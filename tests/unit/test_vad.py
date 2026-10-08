"""Speech segments from synthetic tones and one spoken sentence."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cutter.config import VadConfig, load_profile
from cutter.events import run_audio
from cutter.models import Source, SourcesArtifact, SourcesData, make_meta, write_artifact
from cutter.vad import speech_segments

_TOLERANCE_S = 0.05
_LEAD_S = 0.48
_BURST_S = 0.8
_GAP_S = 0.64
_TAIL_S = 0.48
_MARGIN_DB = 12.0


def test_tone_bursts_match_on_both_backends(tmp_path: Path):
    expected = _expected_bursts()
    silero_path = tmp_path / "bursts.16k.wav"
    # A few samples past a chunk boundary so the short tail is padded.
    _write_bursts(silero_path, 16_000, extra_samples=80)
    energy_path = tmp_path / "bursts.48k.wav"
    _write_bursts(energy_path, 48_000)

    silero = speech_segments(
        silero_path,
        _vad(backend="silero"),
        voice_margin_db=_MARGIN_DB,
    )
    energy = speech_segments(
        energy_path,
        _vad(backend="energy"),
        voice_margin_db=_MARGIN_DB,
    )
    _assert_edges(silero, expected)
    _assert_edges(energy, expected)


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say is not installed")
def test_say_sentence_is_one_silero_segment(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.fail("ffmpeg is required to convert the say clip")
    wav = tmp_path / "sentence.wav"
    _say_to_wav("The cache stores the result and skips the work.", wav)
    spans = speech_segments(wav, load_profile("long").vad, voice_margin_db=_MARGIN_DB)
    assert len(spans) == 1
    start, end = spans[0]
    assert end > start


def test_disabled_vad_records_no_speech(tmp_path: Path):
    _write_project(tmp_path)
    profile = load_profile("long")
    disabled = profile.model_copy(
        update={"vad": profile.vad.model_copy(update={"enabled": False})}
    )
    artifact = run_audio(tmp_path, disabled)
    assert artifact.data.backend == "none"
    assert artifact.data.speech == []
    assert artifact.data.claps == []


def test_run_audio_cache_follows_vad_not_tighten(tmp_path: Path):
    _write_project(tmp_path)
    profile = load_profile("long")
    path = tmp_path / "artifacts" / "audio_events.json"

    first = run_audio(tmp_path, profile)
    first_bytes = path.read_bytes()
    assert first.data.backend == "silero"
    second = run_audio(tmp_path, profile)
    assert path.read_bytes() == first_bytes
    assert second.meta.created_at == first.meta.created_at

    quieter = profile.model_copy(
        update={"vad": profile.vad.model_copy(update={"threshold": 0.2})}
    )
    third = run_audio(tmp_path, quieter)
    assert path.read_bytes() != first_bytes
    assert third.meta.config_hash != first.meta.config_hash

    padded = quieter.model_copy(
        update={"tighten": quieter.tighten.model_copy(update={"pad_head_ms": 10})}
    )
    cached = path.read_bytes()
    fourth = run_audio(tmp_path, padded)
    assert path.read_bytes() == cached
    assert fourth.meta.created_at == third.meta.created_at


def _vad(backend: str) -> VadConfig:
    return load_profile("long").vad.model_copy(
        update={
            "backend": backend,
            "threshold": 0.5,
            "min_speech_ms": 100,
            "min_silence_ms": 200,
            "pad_ms": 0,
        }
    )


def _assert_edges(spans: list[tuple[float, float]], expected: list[tuple[float, float]]) -> None:
    assert len(spans) == len(expected)
    for (start, end), (want_start, want_end) in zip(spans, expected, strict=True):
        assert abs(start - want_start) <= _TOLERANCE_S
        assert abs(end - want_end) <= _TOLERANCE_S
        assert start <= end


def _expected_bursts() -> list[tuple[float, float]]:
    first = (_LEAD_S, _LEAD_S + _BURST_S)
    second_start = first[1] + _GAP_S
    return [first, (second_start, second_start + _BURST_S)]


def _write_bursts(path: Path, sample_rate: int, extra_samples: int = 0) -> None:
    burst = _voiced_burst(_BURST_S, sample_rate)
    signal = np.concatenate(
        [
            np.zeros(int(round(sample_rate * _LEAD_S))),
            burst,
            np.zeros(int(round(sample_rate * _GAP_S))),
            burst,
            np.zeros(int(round(sample_rate * _TAIL_S)) + extra_samples),
        ]
    )
    sf.write(path, signal.astype(np.float32), sample_rate)


def _voiced_burst(duration_s: float, sample_rate: int) -> np.ndarray:
    """Harmonic tone that moves through a few vowels. A pure sine stays under Silero."""
    parts = (
        (0.22, 700, 1200, 125),
        (0.18, 400, 2000, 145),
        (0.22, 500, 1600, 115),
        (0.18, 320, 2200, 155),
    )
    tones = [_vowel(length, f1, f2, f0, sample_rate) for length, f1, f2, f0 in parts]
    burst = np.concatenate(tones)
    target = int(round(sample_rate * duration_s))
    if burst.size < target:
        burst = np.concatenate([burst, np.zeros(target - burst.size)])
    return burst[:target]


def _vowel(
    duration_s: float,
    f1: float,
    f2: float,
    f0: float,
    sample_rate: int,
    amp: float = 0.75,
) -> np.ndarray:
    count = int(round(sample_rate * duration_s))
    times = np.arange(count) / sample_rate
    tone = np.zeros(count)
    for harmonic in range(1, 25):
        freq = f0 * harmonic
        if freq > 7500:
            break
        weight = (
            np.exp(-0.5 * ((freq - f1) / 180) ** 2)
            + 0.7 * np.exp(-0.5 * ((freq - f2) / 250) ** 2)
            + 0.1
        )
        tone += (weight / harmonic) * np.sin(2 * np.pi * freq * times + 0.3 * harmonic)
    peak = float(np.max(np.abs(tone))) + 1e-9
    tone = tone / peak
    attack = int(0.015 * sample_rate)
    envelope = np.ones(count)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-attack:] = np.linspace(1, 0, attack)
    return tone * amp * envelope


def _say_to_wav(text: str, dest: Path) -> None:
    aiff = dest.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(aiff),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
    )


def _write_project(project: Path) -> None:
    audio = project / "artifacts" / "audio"
    audio.mkdir(parents=True)
    _silence_wav(audio / "s01.16k.wav", 16_000)
    _silence_wav(audio / "s01.48k.wav", 48_000)
    write_artifact(
        project / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=1,
                inputs_hash="sha256:in",
                config_hash="sha256:cfg",
            ),
            data=SourcesData(
                fps="30000/1001",
                width=1920,
                height=1080,
                audio_rate=48000,
                audio_channels=1,
                sources=[
                    Source(
                        id="s01",
                        path=str(project / "raw" / "01.mov"),
                        duration_s=1.0,
                        duration_frames=30,
                        start_timecode="00:00:00:00",
                        start_frames=0,
                        vfr_warning=False,
                        asr_wav="artifacts/audio/s01.16k.wav",
                        analysis_wav="artifacts/audio/s01.48k.wav",
                    )
                ],
            ),
        ),
    )


def _silence_wav(path: Path, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate)

"""Clap onsets on synthetic 48 kHz WAVs and the e2e fixture."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cutter.claps import DetectedClap, _merge, _rolling_median, detect_claps
from cutter.config import load_profile
from cutter.events import run_audio
from cutter.models import (
    Source,
    SourcesArtifact,
    SourcesData,
    make_meta,
    write_artifact,
)

PROFILE = load_profile("long")
CLAPS = PROFILE.claps
SAMPLE_RATE = 48_000

# Three 30 ms bursts after the 1 s background window fills. The first pair is
# 150 ms apart, inside the 300 ms gap.
_ONSETS = (1.20, 1.35, 2.10)


def test_close_bursts_merge_into_two_claps(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "bursts.wav", _ONSETS)
    found = detect_claps(wav, [], CLAPS)

    assert len(found) == 2
    assert found[0].t == pytest.approx(_ONSETS[0], abs=0.02)
    assert found[1].t == pytest.approx(_ONSETS[2], abs=0.02)
    assert found[0].t < _ONSETS[1]
    assert found[0].rise_db >= CLAPS.min_rise_db
    assert found[1].rise_db >= CLAPS.min_rise_db


def test_bursts_covered_by_speech_are_dropped(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "covered.wav", _ONSETS)
    found = detect_claps(wav, [(0.0, 3.0)], CLAPS)
    assert found == []


def test_noise_without_bursts_has_no_claps(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "noise.wav", ())
    assert detect_claps(wav, [], CLAPS) == []


def test_merge_keeps_the_first_onset_and_the_louder_rise() -> None:
    merged = _merge(
        [
            DetectedClap(t=1.0, peak_db=-10.0, rise_db=20.0),
            DetectedClap(t=1.15, peak_db=-3.0, rise_db=31.0),
            DetectedClap(t=1.60, peak_db=-6.0, rise_db=24.0),
        ],
        CLAPS.min_gap_ms / 1000,
    )
    assert merged == [
        DetectedClap(t=1.0, peak_db=-3.0, rise_db=31.0),
        DetectedClap(t=1.60, peak_db=-6.0, rise_db=24.0),
    ]


def test_a_long_burst_is_rejected(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "thud.wav", (1.20,), duration_s=0.20)
    assert detect_claps(wav, [], CLAPS) == []


def test_rolling_median_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    level = rng.normal(size=1_500)
    window = 500
    got = _rolling_median(level, window)
    for index in (0, 1, 17, 499, 500, 1_200, 1_499):
        start = max(0, index - window + 1)
        expected = float(np.median(level[start : index + 1]))
        assert got[index] == pytest.approx(expected, abs=1e-9)


def test_audio_stage_records_claps_on_the_source(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "artifacts" / "audio" / "s01.48k.wav", _ONSETS)
    _write_sources(tmp_path, wav)
    artifact = run_audio(tmp_path, PROFILE)

    assert artifact.meta.stage_version == 3
    assert artifact.data.backend == "silero"
    assert artifact.data.speech == []
    assert len(artifact.data.claps) == 2
    assert {clap.source for clap in artifact.data.claps} == {"s01"}
    assert artifact.data.claps[0].t == pytest.approx(_ONSETS[0], abs=0.02)


def test_audio_stage_skips_claps_when_disabled(tmp_path: Path) -> None:
    wav = _write_bursts(tmp_path / "artifacts" / "audio" / "s01.48k.wav", _ONSETS)
    _write_sources(tmp_path, wav)
    profile = PROFILE.model_copy(update={"claps": CLAPS.model_copy(update={"enabled": False})})
    artifact = run_audio(tmp_path, profile)
    assert artifact.data.claps == []


@pytest.mark.skipif(
    shutil.which("say") is None or shutil.which("ffmpeg") is None,
    reason="say and ffmpeg must be on PATH",
)
def test_fixture_onsets_match_claps_json(tmp_path: Path) -> None:
    from tests.e2e.make_fixture import build_project

    project = tmp_path / "fixture"
    build_project(project, clap=True)
    expected = json.loads((project / "claps.json").read_text(encoding="utf-8"))

    # These bursts peak about 13–15 dB over the 1 s speech median, under the
    # profile's min_rise_db of 20. Lowering that key is how a missed clap is
    # recovered; 13 keeps both onsets and drops the say clip.
    tuned = CLAPS.model_copy(update={"min_rise_db": 13})
    first = _extract(project / "raw" / "01.mov", tmp_path / "01.wav")
    found = detect_claps(first, [], tuned)
    assert len(found) == len(expected["01.mov"])
    for hit, onset in zip(found, expected["01.mov"], strict=True):
        assert hit.t == pytest.approx(onset, abs=0.05)

    second = _extract(project / "raw" / "02.mov", tmp_path / "02.wav")
    assert detect_claps(second, [], CLAPS) == []
    assert detect_claps(second, [], tuned) == []


def _write_bursts(path: Path, onsets: tuple[float, ...], *, duration_s: float = 0.03) -> Path:
    """White-noise bursts near -3 dBFS on a -50 dBFS bed."""
    total = int(3.0 * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, _rms(-50.0), total)
    burst_n = int(round(duration_s * SAMPLE_RATE))
    for onset in onsets:
        start = int(round(onset * SAMPLE_RATE))
        samples[start : start + burst_n] += rng.normal(0.0, _rms(-3.0), burst_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), SAMPLE_RATE)
    return path


def _rms(dbfs: float) -> float:
    return 10 ** (dbfs / 20)


def _write_sources(project: Path, analysis: Path) -> None:
    audio = analysis.parent
    asr = audio / "s01.16k.wav"
    if not asr.is_file():
        sf.write(asr, np.zeros(16_000, dtype=np.float32), 16_000)
    write_artifact(
        project / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=1,
                inputs_hash="sha256:sources",
                config_hash="sha256:cfg",
            ),
            data=SourcesData(
                fps="30000/1001",
                width=1920,
                height=1080,
                audio_rate=SAMPLE_RATE,
                audio_channels=1,
                sources=[
                    Source(
                        id="s01",
                        path=str(project / "raw" / "01.mov"),
                        duration_s=3.0,
                        duration_frames=60,
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


def _extract(mov: Path, wav: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(mov),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        check=True,
    )
    return wav

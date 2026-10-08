"""Input and config hashes, and cache hits."""

import os
from datetime import UTC, datetime
from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import load_profile
from cutter.models import Source, SourcesArtifact, SourcesData, make_meta, write_artifact


def test_artifact_bytes_change_the_inputs_hash(tmp_path: Path):
    artifact = tmp_path / "words.json"
    artifact.write_text("one", encoding="utf-8")
    first = inputs_hash(artifacts=[artifact])
    artifact.write_text("two", encoding="utf-8")
    assert inputs_hash(artifacts=[artifact]) != first
    assert first.startswith("sha256:")


def test_media_hash_uses_size_and_mtime_not_bytes(tmp_path: Path):
    media = tmp_path / "01.mov"
    media.write_bytes(b"a" * 10)
    os.utime(media, ns=(1_000_000_000, 1_000_000_000))
    first = inputs_hash(media=[media])

    media.write_bytes(b"b" * 10)
    os.utime(media, ns=(1_000_000_000, 1_000_000_000))
    assert inputs_hash(media=[media]) == first

    os.utime(media, ns=(2_000_000_000, 2_000_000_000))
    assert inputs_hash(media=[media]) != first


def test_config_hash_ignores_sections_the_stage_does_not_read():
    profile = load_profile("long")
    ingest = config_hash(profile, STAGE_CONFIG_SECTIONS["ingest"])
    changed = profile.model_copy(
        update={"judge": profile.judge.model_copy(update={"enabled": False})}
    )
    assert config_hash(changed, STAGE_CONFIG_SECTIONS["ingest"]) == ingest
    assert config_hash(changed, STAGE_CONFIG_SECTIONS["judge"]) != config_hash(
        profile, STAGE_CONFIG_SECTIONS["judge"]
    )


def test_vad_threshold_misses_audio_and_tighten_but_not_retakes():
    profile = load_profile("long")
    audio = config_hash(profile, STAGE_CONFIG_SECTIONS["audio"])
    tighten = config_hash(profile, STAGE_CONFIG_SECTIONS["tighten"])
    retakes = config_hash(profile, STAGE_CONFIG_SECTIONS["retakes"])
    padded = profile.model_copy(
        update={"tighten": profile.tighten.model_copy(update={"pad_head_ms": 10})}
    )
    assert config_hash(padded, STAGE_CONFIG_SECTIONS["audio"]) == audio

    voiced = profile.model_copy(update={"vad": profile.vad.model_copy(update={"threshold": 0.2})})
    assert config_hash(voiced, STAGE_CONFIG_SECTIONS["audio"]) != audio
    assert config_hash(voiced, STAGE_CONFIG_SECTIONS["tighten"]) != tighten
    assert config_hash(voiced, STAGE_CONFIG_SECTIONS["retakes"]) == retakes


def test_cache_hit_requires_stage_version_inputs_and_config(tmp_path: Path):
    path = tmp_path / "sources.json"
    meta = make_meta(
        stage="ingest",
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
        created_at=datetime(2026, 10, 6, tzinfo=UTC),
    )
    write_artifact(
        path,
        SourcesArtifact(
            meta=meta,
            data=SourcesData(
                fps="30000/1001",
                width=1920,
                height=1080,
                audio_rate=48000,
                audio_channels=2,
                sources=[
                    Source(
                        id="s01",
                        path="/tmp/01.mov",
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
    kwargs = {
        "stage": "ingest",
        "stage_version": 1,
        "inputs_hash": "sha256:in",
        "config_hash": "sha256:cfg",
    }
    assert cache_hit(path, **kwargs) is True
    assert cache_hit(path, **{**kwargs, "stage_version": 2}) is False
    assert cache_hit(path, **{**kwargs, "inputs_hash": "sha256:other"}) is False
    assert cache_hit(tmp_path / "missing.json", **kwargs) is False

    path.write_text("{", encoding="utf-8")
    assert cache_hit(path, **kwargs) is False

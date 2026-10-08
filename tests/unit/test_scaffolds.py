"""Empty Phase 2 stages still cache like the Phase 1 stages."""

import logging
from datetime import UTC, datetime

from cutter.align import run_align
from cutter.config import load_profile
from cutter.events import run_audio
from cutter.fillers import run_fillers
from cutter.models import (
    DecisionsArtifact,
    DecisionsData,
    Source,
    SourcesArtifact,
    SourcesData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)

PROFILE = load_profile("long")


def test_audio_is_empty_and_cached(tmp_path):
    _write_sources(tmp_path)
    profile = PROFILE.model_copy(update={"vad": PROFILE.vad.model_copy(update={"enabled": False})})
    first = run_audio(tmp_path, profile)
    second = run_audio(tmp_path, profile)
    assert first.data.backend == "none"
    assert first.data.speech == []
    assert first.data.claps == []
    assert second.meta.created_at == first.meta.created_at
    assert (tmp_path / "artifacts" / "audio_events.json").is_file()


def test_audio_requires_sources(tmp_path):
    try:
        run_audio(tmp_path, PROFILE)
    except FileNotFoundError as exc:
        assert "missing sources artifact" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_align_logs_a_missing_script_once(tmp_path, caplog):
    _write_words(tmp_path)
    with caplog.at_level(logging.INFO, logger="cutter.align"):
        first = run_align(tmp_path, PROFILE)
    script = tmp_path / "script.md"
    assert f"no script at {script}" in caplog.text
    assert first.data.script is None
    assert first.data.sentences == []
    assert first.data.chapters == []
    assert first.data.unscripted == []
    assert first.data.missing == []

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cutter.align"):
        second = run_align(tmp_path, PROFILE)
    assert caplog.text == ""
    assert second.meta.created_at == first.meta.created_at


def test_align_misses_the_cache_when_a_script_appears(tmp_path, caplog):
    _write_words(tmp_path)
    with caplog.at_level(logging.INFO, logger="cutter.align"):
        first = run_align(tmp_path, PROFILE)
    script = tmp_path / "script.md"
    script.write_text("# Cache\n\nHello.\n", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cutter.align"):
        second = run_align(tmp_path, PROFILE)
    assert caplog.text == ""
    assert second.meta.inputs_hash != first.meta.inputs_hash
    assert second.data.script is None
    assert second.data.sentences == []


def test_fillers_are_empty_and_cached(tmp_path):
    _write_words(tmp_path)
    _write_decisions(tmp_path)
    first = run_fillers(tmp_path, PROFILE)
    second = run_fillers(tmp_path, PROFILE)
    assert first.meta.stage == "fillers"
    assert first.data.decisions == []
    assert second.meta.created_at == first.meta.created_at


def _meta(stage: str):
    return make_meta(
        stage=stage,
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
        created_at=datetime(2026, 10, 8, tzinfo=UTC),
    )


def _write_sources(project):
    audio = project / "artifacts" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    (audio / "s01.16k.wav").write_bytes(b"16k")
    (audio / "s01.48k.wav").write_bytes(b"48k")
    write_artifact(
        project / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=_meta("ingest"),
            data=SourcesData(
                fps="30000/1001",
                width=1920,
                height=1080,
                audio_rate=48000,
                audio_channels=1,
                sources=[
                    Source(
                        id="s01",
                        path=str(project / "01.mov"),
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


def _write_words(project):
    write_artifact(
        project / "artifacts" / "words.json",
        WordsArtifact(
            meta=_meta("transcribe"),
            data=WordsData(
                words=[
                    Word(i=0, source="s01", w="Hello.", norm="hello", start=0.1, end=0.4, sent=0)
                ]
            ),
        ),
    )


def _write_decisions(project):
    write_artifact(
        project / "artifacts" / "decisions.json",
        DecisionsArtifact(meta=_meta("judge"), data=DecisionsData(decisions=[])),
    )

"""CLI help and profile overrides."""

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from cutter.cli import app
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

runner = CliRunner()

COMMANDS = [
    "ingest",
    "transcribe",
    "audio",
    "align",
    "retakes",
    "judge",
    "fillers",
    "tighten",
    "export",
    "run",
    "eval",
    "words",
]


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_run_rejects_an_unknown_override(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--set", "ingest.nope=1"])
    assert result.exit_code == 1
    assert "unknown config path" in result.output


def test_run_accepts_a_valid_override_then_stops(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--set", "judge.enabled=false"])
    assert result.exit_code == 1
    assert "unknown config path" not in result.output
    assert "no raw folder" in result.output


def test_eval_set_override(tmp_path):
    result = runner.invoke(app, ["eval", str(tmp_path), "--set", "tighten.max_gap_ms=200"])
    assert result.exit_code == 1
    assert "unknown config path" not in result.output
    assert "missing gold edit" in result.output


def test_judge_requires_words(tmp_path):
    result = runner.invoke(app, ["judge", str(tmp_path), "--no-llm"])
    assert result.exit_code == 1
    assert "missing words artifact" in result.output


def test_tighten_requires_words(tmp_path):
    result = runner.invoke(app, ["tighten", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing words artifact" in result.output


def test_audio_requires_sources(tmp_path):
    result = runner.invoke(app, ["audio", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 1
    assert "missing sources artifact" in result.output


def test_audio_reports_an_empty_scaffold(tmp_path):
    _write_sources(tmp_path)
    result = runner.invoke(app, ["audio", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 0, result.output
    assert "speech segments: 0, claps: 0" in result.output


def test_align_requires_words(tmp_path):
    result = runner.invoke(app, ["align", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 1
    assert "missing words artifact" in result.output


def test_align_reports_no_script(tmp_path):
    _write_words(tmp_path)
    result = runner.invoke(app, ["align", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 0, result.output
    assert "no script" in result.output


def test_fillers_requires_decisions(tmp_path):
    _write_words(tmp_path)
    result = runner.invoke(app, ["fillers", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 1
    assert "missing decisions artifact" in result.output


def test_fillers_reports_an_empty_scaffold(tmp_path):
    _write_words(tmp_path)
    _write_decisions(tmp_path)
    result = runner.invoke(app, ["fillers", str(tmp_path), "--profile", "long"])
    assert result.exit_code == 0, result.output
    assert "filler decisions: 0" in result.output


def _meta(stage: str):
    return make_meta(
        stage=stage,
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
        created_at=datetime(2026, 10, 8, tzinfo=UTC),
    )


def _write_sources(project: Path) -> None:
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


def _write_words(project: Path) -> None:
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


def _write_decisions(project: Path) -> None:
    write_artifact(
        project / "artifacts" / "decisions.json",
        DecisionsArtifact(meta=_meta("judge"), data=DecisionsData(decisions=[])),
    )

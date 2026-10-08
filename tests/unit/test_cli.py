"""CLI help and profile overrides."""

import wave
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from cutter.cli import _echo_summary, _stage_gold_script, app
from cutter.models import (
    DecisionsArtifact,
    DecisionsData,
    Source,
    SourcesArtifact,
    SourcesData,
    TimelineArtifact,
    TimelineData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.run import RunSummary

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


def test_run_summary_prints_phase2_counts(capsys) -> None:
    _echo_summary(
        RunSummary(
            sources=2,
            raw_duration_s=18.4,
            output_duration_s=12.1,
            dropped=4,
            kept=3,
            flagged=1,
            fcpxml=Path("projects/my-video/out/my-video.fcpxml"),
            speech_share=0.64,
            claps=2,
            filler_drops=3,
            script_sentences_found=8,
            script_sentences_missing=1,
            unscripted_spans=0,
            chapters=2,
        )
    )
    out = capsys.readouterr().out
    assert "speech share 0.640, claps 2, filler drops 3" in out
    assert "script sentences found 8, missing 1, unscripted spans 0, chapters 2" in out


def test_gold_script_is_copied_beside_raw_sources(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    project = tmp_path / "sample"
    gold.mkdir()
    project.mkdir()
    script = "# Cache\n\nHello.\n"
    (gold / "script.md").write_text(script, encoding="utf-8")

    _stage_gold_script(gold, project, "script.md")

    dest = project / "script.md"
    assert dest.read_text(encoding="utf-8") == script
    stamped = dest.stat().st_mtime_ns
    _stage_gold_script(gold, project, "script.md")
    assert dest.stat().st_mtime_ns == stamped
    _stage_gold_script(gold, gold, "script.md")
    assert (gold / "script.md").read_text(encoding="utf-8") == script


def test_eval_without_a_gold_script_leaves_the_project_alone(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    project = tmp_path / "sample"
    gold.mkdir()
    project.mkdir()
    _stage_gold_script(gold, project, "script.md")
    assert not (project / "script.md").exists()


def test_gold_script_follows_script_path(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    project = tmp_path / "sample"
    gold.mkdir()
    project.mkdir()
    (gold / "script.md").write_text("# Cache\n\nHello.\n", encoding="utf-8")

    _stage_gold_script(gold, project, "notes/take.md")

    assert (project / "notes" / "take.md").read_text(encoding="utf-8").startswith("# Cache")
    assert not (project / "script.md").exists()


def test_a_broken_fillers_file_does_not_write_the_score(tmp_path: Path, monkeypatch) -> None:
    gold = tmp_path / "gold"
    (gold / "raw").mkdir(parents=True)
    (gold / "manual.fcpxml").write_text("<fcpxml/>", encoding="utf-8")
    meta = make_meta(
        stage="transcribe",
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
        created_at=datetime(2026, 10, 8, tzinfo=UTC),
    )

    def fake_run(project_dir: Path, profile, **_kwargs) -> None:
        artifacts = project_dir / "artifacts"
        artifacts.mkdir(parents=True)
        write_artifact(
            artifacts / "sources.json",
            SourcesArtifact(
                meta=meta.model_copy(update={"stage": "ingest"}),
                data=SourcesData(
                    fps="30000/1001",
                    width=1920,
                    height=1080,
                    audio_rate=48000,
                    audio_channels=1,
                    sources=[
                        Source(
                            id="s01",
                            path="raw/01.mov",
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
        write_artifact(
            artifacts / "words.json",
            WordsArtifact(meta=meta, data=WordsData(words=[])),
        )
        write_artifact(
            artifacts / "timeline.json",
            TimelineArtifact(
                meta=meta.model_copy(update={"stage": "tighten"}),
                data=TimelineData(ranges=[], dropped=[]),
            ),
        )
        (artifacts / "fillers.json").write_text("{", encoding="utf-8")

    def fake_eval(**kwargs):
        kwargs["eval_json"].write_text("{}\n", encoding="utf-8")
        return None, "table"

    monkeypatch.setattr("cutter.cli.run_project", fake_run)
    monkeypatch.setattr("cutter.cli.evaluate_gold", fake_eval)
    result = runner.invoke(app, ["eval", str(gold)])
    assert result.exit_code == 1
    assert not (gold / "eval.json").exists()


def test_words_source_limits_the_window(tmp_path: Path) -> None:
    write_artifact(
        tmp_path / "artifacts" / "words.json",
        WordsArtifact(
            meta=make_meta(
                stage="transcribe",
                stage_version=1,
                inputs_hash="sha256:in",
                config_hash="sha256:cfg",
                created_at=datetime(2026, 10, 8, tzinfo=UTC),
            ),
            data=WordsData(
                words=[
                    Word(i=0, source="s01", w="Hello", norm="hello", start=1.0, end=1.2, sent=0),
                    Word(i=1, source="s02", w="there", norm="there", start=1.1, end=1.3, sent=0),
                ]
            ),
        ),
    )
    both = runner.invoke(app, ["words", str(tmp_path), "--from", "0", "--to", "2"])
    assert both.exit_code == 0
    assert both.stdout.strip() == "Hello 1.000 1.200\nthere 1.100 1.300"
    one = runner.invoke(
        app, ["words", str(tmp_path), "--from", "0", "--to", "2", "--source", "s01"]
    )
    assert one.exit_code == 0
    assert one.stdout.strip() == "Hello 1.000 1.200"


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


def test_audio_reports_no_speech_in_silence(tmp_path):
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
    _silence_wav(audio / "s01.16k.wav", 16_000)
    _silence_wav(audio / "s01.48k.wav", 48_000)
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


def _silence_wav(path: Path, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate)


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

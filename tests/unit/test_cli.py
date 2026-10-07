"""CLI help and profile overrides."""

from typer.testing import CliRunner

from cutter.cli import app

runner = CliRunner()

COMMANDS = [
    "ingest",
    "transcribe",
    "retakes",
    "judge",
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

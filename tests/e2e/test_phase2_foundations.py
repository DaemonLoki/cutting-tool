"""Phase 2 scaffolds on the extended synthetic fixture. Parakeet needs a Metal GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.e2e.make_fixture import build_project
from tests.e2e.test_pipeline import _metal_available
from typer.testing import CliRunner

from cutter.cli import app
from cutter.models import AlignmentArtifact, AudioEventsArtifact, DecisionsArtifact

runner = CliRunner()

pytestmark = pytest.mark.skipif(not _metal_available(), reason="Parakeet needs a Metal GPU")

_STAGES = (
    "sources.json",
    "words.json",
    "audio_events.json",
    "alignment.json",
    "decisions.json",
    "fillers.json",
    "timeline.json",
    "export.json",
)


def test_scaffolds_are_empty_and_a_second_run_skips_every_stage(tmp_path: Path) -> None:
    project = tmp_path / "synthetic"
    build_project(project, fillers=True, clap=True, script=True)

    result = runner.invoke(app, ["run", str(project), "--no-llm"])
    assert result.exit_code == 0, result.output

    artifacts = project / "artifacts"
    audio = AudioEventsArtifact.model_validate_json(
        (artifacts / "audio_events.json").read_text(encoding="utf-8")
    )
    alignment = AlignmentArtifact.model_validate_json(
        (artifacts / "alignment.json").read_text(encoding="utf-8")
    )
    fillers = DecisionsArtifact.model_validate_json(
        (artifacts / "fillers.json").read_text(encoding="utf-8")
    )
    assert audio.data.backend == "none"
    assert audio.data.speech == []
    assert audio.data.claps == []
    assert alignment.data.script is None
    assert alignment.data.chapters == []
    assert alignment.data.sentences == []
    assert alignment.data.unscripted == []
    assert alignment.data.missing == []
    assert fillers.data.decisions == []

    created = {name: (artifacts / name).read_bytes() for name in _STAGES}
    again = runner.invoke(app, ["run", str(project), "--no-llm"])
    assert again.exit_code == 0, again.output
    assert {name: (artifacts / name).read_bytes() for name in _STAGES} == created

"""Phase 2 stages on the extended synthetic fixture. Parakeet needs a Metal GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.e2e.make_fixture import build_project
from tests.e2e.test_pipeline import _metal_available
from typer.testing import CliRunner

from cutter.cli import app
from cutter.models import (
    AlignmentArtifact,
    AudioEventsArtifact,
    DecisionsArtifact,
    WordsArtifact,
)

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


def test_speech_alignment_and_fillers_and_a_second_run_is_cached(tmp_path: Path) -> None:
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
    assert audio.data.backend == "silero"
    assert audio.data.speech
    assert audio.data.claps == []
    assert alignment.data.script is not None
    assert alignment.data.script.path == "script.md"
    assert [chapter.title for chapter in alignment.data.chapters] == ["Cache", "Agent"]
    assert len(alignment.data.sentences) == 3
    assert len(alignment.data.missing) == 1
    assert all(decision.kind == "filler" for decision in fillers.data.decisions)
    words = WordsArtifact.model_validate_json(
        (artifacts / "words.json").read_text(encoding="utf-8")
    )
    spoken = {word.norm for word in words.data.words}
    dropped_norms: set[str] = set()
    for decision in fillers.data.decisions:
        if decision.action != "drop":
            continue
        first, last = decision.dropped_words
        dropped_norms.update(word.norm for word in words.data.words if first <= word.i <= last)
    # The fixture says "Um" and "uh" in the kept sentence. Tighten does not
    # remove them from the rough cut yet; this checks the filler decisions.
    assert {"um", "uh"} <= spoken
    assert {"um", "uh"} <= dropped_norms

    created = {name: (artifacts / name).read_bytes() for name in _STAGES}
    again = runner.invoke(app, ["run", str(project), "--no-llm"])
    assert again.exit_code == 0, again.output
    assert {name: (artifacts / name).read_bytes() for name in _STAGES} == created

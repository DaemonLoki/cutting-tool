"""Full pipeline on speech generated with ``say``. Parakeet needs a Metal GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.e2e.make_fixture import EXPECTED, build_project
from typer.testing import CliRunner

from cutter.cli import app
from cutter.models import TimelineArtifact, WordsArtifact

runner = CliRunner()


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except Exception:
        return False
    try:
        mx.zeros((1,))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _metal_available(), reason="Parakeet needs a Metal GPU")


def test_pipeline_drops_the_failed_takes(tmp_path: Path) -> None:
    project = tmp_path / "synthetic"
    build_project(project)

    result = runner.invoke(app, ["run", str(project), "--no-llm"])

    assert result.exit_code == 0, result.output
    fcpxml = project / "out" / "synthetic.fcpxml"
    assert fcpxml.is_file()
    assert _kept_text(project) == _normalize(EXPECTED)


def _kept_text(project: Path) -> str:
    artifacts = project / "artifacts"
    words = WordsArtifact.model_validate_json(
        (artifacts / "words.json").read_text(encoding="utf-8")
    )
    timeline = TimelineArtifact.model_validate_json(
        (artifacts / "timeline.json").read_text(encoding="utf-8")
    )
    kept: list[str] = []
    for word in words.data.words:
        midpoint = (word.start + word.end) / 2
        if any(
            item.source == word.source and item.in_s <= midpoint < item.out_s
            for item in timeline.data.ranges
        ):
            kept.append(word.norm or _normalize(word.w))
    return " ".join(kept)


def _normalize(text: str) -> str:
    tokens = []
    for raw in text.split():
        token = "".join(character for character in raw.casefold() if character.isalnum())
        if token:
            tokens.append(token)
    return " ".join(tokens)

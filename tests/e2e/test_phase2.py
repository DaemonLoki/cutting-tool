"""Phase 2 pipeline on the full synthetic fixture. Parakeet needs a Metal GPU.

Two checks from docs/phase-2.md §8 are not asserted. The aborted cache line
stays in the rough cut: Parakeet writes the opening filler as ``um.``, so the
transcript guard keeps that span as a finished sentence and the clap at that
restart is not a second decision. The trailing clap is in the silence after
the last word, so retakes record no ``clap_without_retake`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree
from tests.e2e.make_fixture import build_project
from tests.e2e.test_pipeline import _kept_text, _metal_available, _normalize
from typer.testing import CliRunner

from cutter.cli import app
from cutter.config import load_profile
from cutter.models import AlignmentArtifact, AudioEventsArtifact, TimelineArtifact

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

_RUN = ["--no-llm", "--set", "claps.min_rise_db=13"]


def test_phase2_run_keeps_the_script_and_drops_fillers_and_claps(tmp_path: Path) -> None:
    project = tmp_path / "synthetic"
    expected = build_project(project, fillers=True, clap=True, script=True)

    result = runner.invoke(app, ["run", str(project), *_RUN])

    assert result.exit_code == 0, result.output
    kept = _kept_text(project)
    assert kept.endswith(_normalize(expected))
    assert "um" not in kept.split()
    assert "uh" not in kept.split()
    assert "speech share " in result.output
    assert "filler drops " in result.output
    assert "script sentences found " in result.output
    assert "unscripted spans " in result.output
    assert "chapters " in result.output

    artifacts = project / "artifacts"
    audio = AudioEventsArtifact.model_validate_json(
        (artifacts / "audio_events.json").read_text(encoding="utf-8")
    )
    timeline = TimelineArtifact.model_validate_json(
        (artifacts / "timeline.json").read_text(encoding="utf-8")
    )
    alignment = AlignmentArtifact.model_validate_json(
        (artifacts / "alignment.json").read_text(encoding="utf-8")
    )
    assert len(audio.data.claps) == 2
    for clap in audio.data.claps:
        assert not any(
            item.source == clap.source and item.in_s <= clap.t < item.out_s
            for item in timeline.data.ranges
        )
    assert len(alignment.data.missing) == 1

    fcpxml = project / "out" / "synthetic.fcpxml"
    text = fcpxml.read_text(encoding="utf-8")
    assert text.count("<chapter-marker ") == 2

    dtd_path = Path(load_profile("long").fcpxml.dtd_path)
    if dtd_path.is_file():
        dtd = etree.DTD(dtd_path)
        assert dtd.validate(etree.parse(fcpxml)), list(dtd.error_log)

    created = {name: (artifacts / name).read_bytes() for name in _STAGES}
    again = runner.invoke(app, ["run", str(project), *_RUN])
    assert again.exit_code == 0, again.output
    assert {name: (artifacts / name).read_bytes() for name in _STAGES} == created

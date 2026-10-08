"""VFR fixture. The 3-minute build is opt-in; a 4-second file runs by default."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from tests.e2e.make_vfr_fixture import build_project, build_tiny, probe_video
from typer.testing import CliRunner, Result

from cutter.cli import app

runner = CliRunner()


def _stderr(result: Result) -> str:
    """The warning is printed on stderr. Some runners also copy it into output."""
    if result.stderr:
        return result.stderr
    return result.output

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def test_tiny_vfr_source_prints_the_ingest_warning(tmp_path: Path) -> None:
    project = tmp_path / "tiny-vfr"
    media = build_tiny(project)
    r_rate, avg_rate, _duration = probe_video(media)
    assert r_rate != avg_rate

    result = runner.invoke(app, ["ingest", str(project)])

    assert result.exit_code == 0, result.output
    captured = _stderr(result)
    assert "VFR warning" in captured
    assert media.name in captured
    assert "avg_frame_rate" in captured
    assert "r_frame_rate" in captured


@pytest.mark.skipif(
    os.environ.get("CUTTER_VFR") != "1",
    reason="set CUTTER_VFR=1 to build the 3-minute VFR source",
)
def test_long_vfr_fixture_prints_the_ingest_warning(tmp_path: Path) -> None:
    project = tmp_path / "vfr"
    media = build_project(project)
    r_rate, avg_rate, duration = probe_video(media)
    assert r_rate != avg_rate
    assert duration == pytest.approx(180, abs=2)

    result = runner.invoke(app, ["ingest", str(project)])

    assert result.exit_code == 0, result.output
    captured = _stderr(result)
    assert "VFR warning" in captured
    assert media.name in captured

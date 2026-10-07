"""Primary-storyline spans from a gold FCPXML."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import pytest

from cutter.fcpxml_parse import SourceSpan, parse_fcpxml

_SRC = "file:///Users/stefanblos/Videos/my%20clip.MP4"


def _write(
    tmp_path: Path,
    spine: str,
    *,
    asset_start: str = "10s",
    src: str = _SRC,
    extra_project: str = "",
) -> Path:
    path = tmp_path / "Info.fcpxml"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.14">
  <resources>
    <asset id="r2" name="clip" start="{asset_start}" duration="30s" hasVideo="1" hasAudio="1">
      <media-rep kind="original-media" src="{src}"/>
    </asset>
  </resources>
  <library>
    <event name="day">
      <project name="demo">
        <sequence duration="10s" tcStart="0s">
          <spine>
            {spine}
          </spine>
        </sequence>
      </project>
      {extra_project}
    </event>
  </library>
</fcpxml>
""",
        encoding="utf-8",
    )
    return path


def test_asset_clip_is_file_relative(tmp_path: Path):
    path = _write(
        tmp_path,
        '<asset-clip ref="r2" offset="0s" name="clip" start="15s" duration="5s"/>',
    )
    result = parse_fcpxml(path)
    assert result.spans == [SourceSpan("my clip.MP4", 5.0, 10.0)]
    assert result.skipped == []


def test_sync_clip_is_skipped_and_warned(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    spine = """
            <sync-clip ref="r2" offset="0s" start="0s" duration="1s"/>
            <mc-clip ref="r2" offset="1s" start="0s" duration="1s"/>
            <ref-clip ref="r4" offset="2s" name="compound" duration="1s"/>
            <asset-clip ref="r2" offset="3s" name="clip" start="15s" duration="5s"/>
    """
    path = _write(tmp_path, spine)
    with caplog.at_level(logging.WARNING, logger="cutter.fcpxml_parse"):
        result = parse_fcpxml(path)
    assert result.spans == [SourceSpan("my clip.MP4", 5.0, 10.0)]
    assert result.skipped == ["sync-clip", "mc-clip", "ref-clip"]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "sync-clip" in caplog.text


def test_gap_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    spine = """
            <gap name="Gap" offset="0s" start="0s" duration="2s"/>
            <asset-clip ref="r2" offset="2s" name="clip" start="15s" duration="5s">
              <asset-clip ref="r2" offset="0s" start="0s" duration="1s"/>
              <title name="Title" start="15s" duration="1s"/>
            </asset-clip>
    """
    path = _write(tmp_path, spine)
    with caplog.at_level(logging.WARNING, logger="cutter.fcpxml_parse"):
        result = parse_fcpxml(path)
    assert result.spans == [SourceSpan("my clip.MP4", 5.0, 10.0)]
    assert result.skipped == []
    assert caplog.records == []


def test_zero_duration_parses(tmp_path: Path):
    path = _write(
        tmp_path,
        '<asset-clip ref="r2" offset="0s" name="clip" start="0s" duration="0s"/>',
        asset_start="0s",
    )
    result = parse_fcpxml(path)
    assert result.spans == [SourceSpan("my clip.MP4", 0.0, 0.0)]


def test_rational_and_decimal_times(tmp_path: Path):
    spine = """
            <asset-clip ref="r2" offset="0s" start="2002/24000s" duration="1001/24000s"/>
            <asset-clip ref="r2" offset="1s" start="12.5s" duration="1s"/>
    """
    path = _write(tmp_path, spine, asset_start="1001/24000s")
    result = parse_fcpxml(path)
    asset_start = Fraction(1001, 24000)
    first = Fraction(2002, 24000) - asset_start
    second = Fraction("12.5") - asset_start
    assert result.spans == [
        SourceSpan("my clip.MP4", float(first), float(first + Fraction(1001, 24000))),
        SourceSpan("my clip.MP4", float(second), float(second + 1)),
    ]


def test_clip_uses_nested_ref_and_prefers_its_own_times(tmp_path: Path):
    spine = """
            <clip offset="0s" name="uses-clip" start="15s" duration="5s">
              <video ref="r2" offset="0s" start="0s" duration="1s"/>
            </clip>
            <clip offset="5s" name="uses-nested">
              <audio ref="r2" offset="0s" start="20s" duration="2s"/>
            </clip>
    """
    path = _write(tmp_path, spine)
    result = parse_fcpxml(path)
    assert result.spans == [
        SourceSpan("my clip.MP4", 5.0, 10.0),
        SourceSpan("my clip.MP4", 10.0, 12.0),
    ]


def test_second_project_is_ignored(tmp_path: Path):
    extra = """
      <project name="rejects">
        <sequence duration="1s" tcStart="0s">
          <spine>
            <asset-clip ref="r2" offset="0s" name="clip" start="0s" duration="1s"/>
          </spine>
        </sequence>
      </project>
    """
    path = _write(
        tmp_path,
        '<asset-clip ref="r2" offset="0s" name="clip" start="15s" duration="5s"/>',
        extra_project=extra,
    )
    result = parse_fcpxml(path)
    assert result.spans == [SourceSpan("my clip.MP4", 5.0, 10.0)]


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_fcpxml(tmp_path / "missing.fcpxml")


def test_missing_project_and_spine_raise(tmp_path: Path):
    bare = tmp_path / "bare.fcpxml"
    bare.write_text('<fcpxml version="1.14"><resources/></fcpxml>', encoding="utf-8")
    with pytest.raises(ValueError, match="no project"):
        parse_fcpxml(bare)

    no_spine = tmp_path / "no-spine.fcpxml"
    no_spine.write_text(
        """<fcpxml version="1.14">
          <library><event><project name="demo"><sequence/></project></event></library>
        </fcpxml>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no spine"):
        parse_fcpxml(no_spine)

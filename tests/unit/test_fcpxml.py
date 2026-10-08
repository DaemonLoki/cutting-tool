"""FCPXML export: golden snapshot, rational time, and DTD validation."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lxml import etree
from typer.testing import CliRunner

from cutter.cli import app
from cutter.config import Profile, load_profile
from cutter.fcpxml import (
    REFERENCE_COLOR_SPACE,
    FcpxmlValidationError,
    build_fcpxml,
    export_fcpxml,
)
from cutter.models import (
    DroppedSpan,
    Marker,
    Range,
    Source,
    SourcesArtifact,
    SourcesData,
    TimelineArtifact,
    TimelineChapter,
    TimelineData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic-rough-cut.fcpxml"
GOLDEN_CHAPTERS = (
    Path(__file__).resolve().parents[1] / "fixtures" / "synthetic-rough-cut-chapters.fcpxml"
)
DTD_PATH = Path(load_profile().fcpxml.dtd_path)

FPS = "24000/1001"
PROJECT = "demo"
SOURCE_PATH = "/sources/Näme clip.mov"
SOURCE_URI = "file:///sources/N%C3%A4me%20clip.mov"
START_FRAMES = 1001
IN_FRAME = 10
OUT_FRAME = 34
WORD_INDEX = 7
WORD_START_S = 0.5
SECOND_IN = 3
SECOND_OUT = 18
DROPPED_IN_S = 1.0
DROPPED_OUT_S = 2.5
DECISION_ID = "d003"
CHECK_TEXT = "CHECK: low-confidence retake (d003)"
CHAPTER_WORD = 20
CHAPTER_WORD_START_S = 0.2
FILLER_IN_S = 3.0
FILLER_OUT_S = 3.25
FILLER_ID = "f003"

_TIME = re.compile(r"^(?:0s|\d+s|\d+/\d+s)$")
_TIME_ATTRS = {"frameDuration", "start", "duration", "offset", "tcStart"}
_MISSING = object()
_KNOWN_FORMAT_NAMES = {
    "FFVideoFormat3840x2160p2398",
    "FFVideoFormat7680x4320p60",
    "FFVideoFormat3840x2160p60",
    "FFVideoFormat3840x2160p30",
}


def _t(fps: str, frames: int) -> str:
    rate = Fraction(fps)
    frame_dur = Fraction(rate.denominator, rate.numerator)
    value = Fraction(frames) * frame_dur
    if value == 0:
        return "0s"
    return f"{value.numerator}/{value.denominator}s"


def _floor_frames(seconds: float, fps: str) -> int:
    return math.floor(Fraction(seconds) * Fraction(fps))


def _ceil_frames(seconds: float, fps: str) -> int:
    return math.ceil(Fraction(seconds) * Fraction(fps))


def _parse_time(value: str) -> Fraction:
    assert "." not in value, value
    assert _TIME.fullmatch(value), value
    body = value[:-1]
    if "/" not in body:
        return Fraction(int(body))
    numerator, denominator = body.split("/")
    return Fraction(int(numerator), int(denominator))


def _root(xml: str) -> etree._Element:
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    return etree.fromstring(xml.encode("utf-8"), parser)


def _source(
    source_id: str,
    path: str,
    *,
    duration_frames: int,
    start_frames: int,
    duration_s: float = 10.0,
) -> Source:
    return Source(
        id=source_id,
        path=path,
        duration_s=duration_s,
        duration_frames=duration_frames,
        start_timecode="01:00:00:00" if start_frames else "00:00:00:00",
        start_frames=start_frames,
        vfr_warning=False,
        asr_wav=f"artifacts/audio/{source_id}.16k.wav",
        analysis_wav=f"artifacts/audio/{source_id}.48k.wav",
    )


def _sources(
    fps: str = FPS,
    width: int = 3840,
    height: int = 2160,
    sources: list[Source] | None = None,
    audio_rate: int = 48000,
    audio_channels: int = 2,
) -> SourcesData:
    if sources is None:
        sources = [
            _source("s01", SOURCE_PATH, duration_frames=5000, start_frames=START_FRAMES),
            _source("s02", "/sources/second.mov", duration_frames=800, start_frames=0),
        ]
    return SourcesData(
        fps=fps,
        width=width,
        height=height,
        audio_rate=audio_rate,
        audio_channels=audio_channels,
        sources=sources,
    )


def _timeline() -> TimelineData:
    return TimelineData(
        ranges=[
            Range(
                id="keep-1",
                source="s01",
                in_s=0.4,
                out_s=1.5,
                in_frame=IN_FRAME,
                out_frame=OUT_FRAME,
                first_word=WORD_INDEX,
                last_word=WORD_INDEX,
                markers=[Marker(at_word=WORD_INDEX, text=CHECK_TEXT)],
            ),
            Range(
                id="keep-2",
                source="s02",
                in_s=0.1,
                out_s=0.8,
                in_frame=SECOND_IN,
                out_frame=SECOND_OUT,
                first_word=0,
                last_word=0,
            ),
        ],
        dropped=[
            DroppedSpan(
                source="s01",
                in_s=DROPPED_IN_S,
                out_s=DROPPED_OUT_S,
                decision=DECISION_ID,
            )
        ],
    )


def _word_starts() -> dict[int, float]:
    return {WORD_INDEX: WORD_START_S}


def _chapter_word_starts() -> dict[int, float]:
    return {WORD_INDEX: WORD_START_S, CHAPTER_WORD: CHAPTER_WORD_START_S}


def _chapters_timeline() -> TimelineData:
    base = _timeline()
    second = base.ranges[1].model_copy(
        update={"first_word": CHAPTER_WORD, "last_word": CHAPTER_WORD}
    )
    return TimelineData(
        ranges=[base.ranges[0], second],
        dropped=[
            *base.dropped,
            DroppedSpan(
                source="s01",
                in_s=FILLER_IN_S,
                out_s=FILLER_OUT_S,
                decision=FILLER_ID,
            ),
        ],
        chapters=[
            TimelineChapter(id="c01", title="Cache", at_word=WORD_INDEX),
            TimelineChapter(id="c02", title="Agent", at_word=CHAPTER_WORD),
        ],
    )


def _fillers_profile() -> Profile:
    profile = load_profile()
    return profile.model_copy(
        update={"fcpxml": profile.fcpxml.model_copy(update={"rejects_include_fillers": True})}
    )


def _profile(**fcpxml: str) -> Profile:
    profile = load_profile()
    if not fcpxml:
        return profile
    return profile.model_copy(update={"fcpxml": profile.fcpxml.model_copy(update=fcpxml)})


def _build(
    sources: SourcesData | None = None,
    timeline: TimelineData | None = None,
    word_starts: dict[int, float] | None | object = _MISSING,
    profile: Profile | None = None,
) -> str:
    if word_starts is _MISSING:
        resolved: dict[int, float] | None = _word_starts()
    elif word_starts is None or isinstance(word_starts, dict):
        resolved = word_starts
    else:
        raise AssertionError(word_starts)
    return build_fcpxml(
        sources if sources is not None else _sources(),
        timeline if timeline is not None else _timeline(),
        profile if profile is not None else load_profile(),
        PROJECT,
        resolved,
    )


def test_snapshot_matches_golden():
    profile = load_profile()
    assert "\u2013" in profile.fcpxml.project_name

    xml = _build()
    root = _root(xml)
    assert root.get("version") == "1.14"

    fmt = root.find("resources/format")
    assert fmt is not None
    assert fmt.get("name") == "FFVideoFormat3840x2160p2398"
    assert fmt.get("frameDuration") == "1001/24000s"
    assert fmt.get("width") == "3840"
    assert fmt.get("height") == "2160"
    assert fmt.get("colorSpace") == "1-1-1 (Rec. 709)"

    assets = root.findall("resources/asset")
    assert len(assets) == 2
    camera = assets[0]
    assert camera.get("hasVideo") == "1"
    assert camera.get("hasAudio") == "1"
    assert camera.get("videoSources") == "1"
    assert camera.get("audioSources") == "1"
    assert camera.get("audioChannels") == "2"
    assert camera.get("audioRate") == "48000"
    assert camera.get("format") == "r1"
    assert camera.get("start") == _t(FPS, START_FRAMES)
    assert "uid" not in camera.attrib
    media = camera.find("media-rep")
    assert media is not None
    assert media.get("kind") == "original-media"
    assert media.get("src") == SOURCE_URI
    assert media.get("sig") is None
    assert media.find("bookmark") is None
    assert camera.find("metadata") is None
    assert root.find(".//keyword") is None

    projects = root.findall("library/event/project")
    assert [project.get("name") for project in projects] == [
        profile.fcpxml.project_name.replace("{project}", PROJECT),
        profile.fcpxml.rejects_project_name.replace("{project}", PROJECT),
    ]
    assert root.find("library/event").get("name") == profile.fcpxml.event_name.replace(
        "{project}", PROJECT
    )

    rough = projects[0].findall("sequence/spine/asset-clip")
    word_frame = _floor_frames(WORD_START_S, FPS)
    assert IN_FRAME <= word_frame < OUT_FRAME
    assert rough[0].get("start") == _t(FPS, START_FRAMES + IN_FRAME)
    assert rough[0].get("duration") == _t(FPS, OUT_FRAME - IN_FRAME)
    assert rough[0].get("offset") == "0s"
    assert rough[0].get("format") is None
    assert rough[0].get("tcFormat") == "NDF"
    assert rough[0].get("audioRole") == profile.fcpxml.audio_role
    marker = rough[0].find("marker")
    assert marker is not None
    assert marker.get("start") == _t(FPS, START_FRAMES + word_frame)
    assert marker.get("duration") == _t(FPS, 1)
    assert marker.get("value") == CHECK_TEXT
    assert rough[1].get("offset") == _t(FPS, OUT_FRAME - IN_FRAME)
    assert rough[1].get("start") == _t(FPS, SECOND_IN)
    assert rough[1].get("duration") == _t(FPS, SECOND_OUT - SECOND_IN)

    sequence = projects[0].find("sequence")
    assert sequence is not None
    assert sequence.get("duration") == _t(FPS, (OUT_FRAME - IN_FRAME) + (SECOND_OUT - SECOND_IN))
    assert sequence.get("audioRate") == "48k"
    assert sequence.get("audioLayout") == "stereo"
    assert sequence.get("tcStart") == "0s"
    assert sequence.get("tcFormat") == "NDF"

    dropped_in = _floor_frames(DROPPED_IN_S, FPS)
    dropped_out = _ceil_frames(DROPPED_OUT_S, FPS)
    rejects = projects[1].findall("sequence/spine/asset-clip")
    assert len(rejects) == 1
    assert rejects[0].get("start") == _t(FPS, START_FRAMES + dropped_in)
    assert rejects[0].get("duration") == _t(FPS, dropped_out - dropped_in)
    reject_marker = rejects[0].find("marker")
    assert reject_marker is not None
    assert reject_marker.get("value") == f"{DECISION_ID}: retake"
    assert reject_marker.get("start") == rejects[0].get("start")
    assert reject_marker.get("duration") == _t(FPS, 1)

    assert xml.encode("utf-8") == GOLDEN.read_bytes()


def test_excluded_filler_keeps_the_phase1_snapshot():
    timeline = _timeline().model_copy(
        update={
            "dropped": [
                *_timeline().dropped,
                DroppedSpan(
                    source="s01",
                    in_s=FILLER_IN_S,
                    out_s=FILLER_OUT_S,
                    decision=FILLER_ID,
                ),
            ],
            "chapters": [TimelineChapter(id="c01", title="Missing", at_word=99)],
        }
    )
    xml = _build(timeline=timeline)
    assert _root(xml).find(".//chapter-marker") is None
    assert f"{FILLER_ID}: filler" not in xml
    assert xml.encode("utf-8") == GOLDEN.read_bytes()


def test_snapshot_chapters_and_filler_reject():
    xml = _build(
        timeline=_chapters_timeline(),
        word_starts=_chapter_word_starts(),
        profile=_fillers_profile(),
    )
    root = _root(xml)
    projects = root.findall("library/event/project")
    rough = projects[0].findall("sequence/spine/asset-clip")
    assert [child.tag for child in rough[0]] == ["marker", "chapter-marker"]
    chapter = rough[0].find("chapter-marker")
    assert chapter is not None
    word_frame = _floor_frames(WORD_START_S, FPS)
    assert chapter.get("start") == _t(FPS, START_FRAMES + word_frame)
    assert chapter.get("value") == "Cache"
    assert chapter.get("duration") is None
    assert chapter.get("posterOffset") is None
    assert chapter.get("note") is None

    agent = rough[1].find("chapter-marker")
    assert agent is not None
    assert [child.tag for child in rough[1]] == ["chapter-marker"]
    agent_frame = _floor_frames(CHAPTER_WORD_START_S, FPS)
    assert SECOND_IN <= agent_frame < SECOND_OUT
    assert agent.get("start") == _t(FPS, agent_frame)
    assert agent.get("value") == "Agent"
    assert agent.get("duration") is None
    assert agent.get("posterOffset") is None

    rejects = projects[1].findall("sequence/spine/asset-clip")
    assert len(rejects) == 2
    retake = rejects[0].find("marker")
    filler = rejects[1].find("marker")
    assert retake is not None and filler is not None
    assert retake.get("value") == f"{DECISION_ID}: retake"
    assert filler.get("value") == f"{FILLER_ID}: filler"
    filler_in = _floor_frames(FILLER_IN_S, FPS)
    filler_out = _ceil_frames(FILLER_OUT_S, FPS)
    assert filler.get("start") == _t(FPS, START_FRAMES + filler_in)
    assert rejects[1].get("duration") == _t(FPS, filler_out - filler_in)
    assert rejects[1].get("offset") == rejects[0].get("duration")
    media = root.find("resources/asset/media-rep")
    assert media is not None
    assert media.get("kind") == "original-media"

    assert xml.encode("utf-8") == GOLDEN_CHAPTERS.read_bytes()


def test_filler_rejects_stay_out_until_enabled():
    xml = _build(timeline=_chapters_timeline(), word_starts=_chapter_word_starts())
    root = _root(xml)
    values = [marker.get("value") for marker in root.findall(".//marker")]
    assert values == [CHECK_TEXT, f"{DECISION_ID}: retake"]
    assert [marker.get("value") for marker in root.findall(".//chapter-marker")] == [
        "Cache",
        "Agent",
    ]
    projects = root.findall("library/event/project")
    assert len(projects[1].findall("sequence/spine/asset-clip")) == 1


def test_dtd_validates_chapters_golden(tmp_path: Path):
    if not DTD_PATH.is_file():
        pytest.skip(f"FCPXML DTD not found at {DTD_PATH}")
    xml = _build(
        timeline=_chapters_timeline(),
        word_starts=_chapter_word_starts(),
        profile=_fillers_profile(),
    )
    assert xml.encode("utf-8") == GOLDEN_CHAPTERS.read_bytes()
    result = export_fcpxml(
        _sources(),
        _chapters_timeline(),
        _fillers_profile(),
        PROJECT,
        tmp_path / f"{PROJECT}.fcpxml",
        _chapter_word_starts(),
    )
    assert result.dtd_validated is True
    assert result.warning is None
    assert result.path.read_bytes() == GOLDEN_CHAPTERS.read_bytes()
    dtd = etree.DTD(DTD_PATH)
    assert dtd.validate(etree.parse(result.path)), list(dtd.error_log)
    assert dtd.validate(etree.parse(GOLDEN_CHAPTERS)), list(dtd.error_log)


def test_marker_without_word_time_uses_clip_in_point():
    xml = _build(word_starts=None)
    marker = _root(xml).find("library/event/project/sequence/spine/asset-clip/marker")
    assert marker is not None
    assert marker.get("start") == _t(FPS, START_FRAMES + IN_FRAME)

    outside = _build(word_starts={WORD_INDEX: 0.0})
    clamped = _root(outside).find("library/event/project/sequence/spine/asset-clip/marker")
    assert clamped is not None
    assert _floor_frames(0.0, FPS) < IN_FRAME
    assert clamped.get("start") == _t(FPS, START_FRAMES + IN_FRAME)


@pytest.mark.parametrize(
    ("width", "height", "fps", "name", "frame_duration"),
    [
        (3840, 2160, "24000/1001", "FFVideoFormat3840x2160p2398", "1001/24000s"),
        (3840, 2160, "30/1", "FFVideoFormat3840x2160p30", "1/30s"),
        (3840, 2160, "60/1", "FFVideoFormat3840x2160p60", "1/60s"),
        (7680, 4320, "60/1", "FFVideoFormat7680x4320p60", "1/60s"),
        (1920, 1080, "24000/1001", None, "1001/24000s"),
        (3840, 2160, "30000/1001", None, "1001/30000s"),
    ],
)
def test_format_name_follows_reference_geometry(width, height, fps, name, frame_duration):
    sources = _sources(
        fps=fps,
        width=width,
        height=height,
        sources=[_source("s01", SOURCE_PATH, duration_frames=100, start_frames=0)],
    )
    timeline = TimelineData(ranges=[], dropped=[])
    fmt = _root(_build(sources, timeline, word_starts={})).find("resources/format")
    assert fmt is not None
    assert fmt.get("name") == name
    assert fmt.get("frameDuration") == frame_duration
    assert fmt.get("width") == str(width)
    assert fmt.get("height") == str(height)
    assert fmt.get("colorSpace") == REFERENCE_COLOR_SPACE


def _frame_dur(fps: str) -> Fraction:
    rate = Fraction(fps)
    return Fraction(rate.denominator, rate.numerator)


@st.composite
def _documents(draw):
    fps = draw(st.sampled_from(["24000/1001", "30000/1001", "24/1", "30/1", "60/1", "60000/1001"]))
    width, height = draw(st.sampled_from([(3840, 2160), (1920, 1080), (1280, 720), (7680, 4320)]))
    count = draw(st.integers(min_value=1, max_value=3))
    sources: list[Source] = []
    for index in range(count):
        duration_frames = draw(st.integers(min_value=2, max_value=400))
        start_frames = draw(st.integers(min_value=0, max_value=5000))
        sources.append(
            _source(
                f"s{index}",
                f"/sources/clip {index} ä.mov",
                duration_frames=duration_frames,
                start_frames=start_frames,
            )
        )
    range_count = draw(st.integers(min_value=1, max_value=5))
    ranges: list[Range] = []
    word_starts: dict[int, float] = {}
    for index in range(range_count):
        source = draw(st.sampled_from(sources))
        in_frame = draw(st.integers(min_value=0, max_value=source.duration_frames - 1))
        out_frame = draw(st.integers(min_value=in_frame + 1, max_value=source.duration_frames))
        markers: list[Marker] = []
        if draw(st.booleans()):
            word = 1000 + index
            markers.append(Marker(at_word=word, text=f"CHECK: note {index}"))
            if draw(st.booleans()):
                word_starts[word] = draw(
                    st.floats(min_value=0, max_value=30, allow_nan=False, allow_infinity=False)
                )
        ranges.append(
            Range(
                id=f"keep-{index}",
                source=source.id,
                in_s=0.0,
                out_s=1.0,
                in_frame=in_frame,
                out_frame=out_frame,
                first_word=0,
                last_word=0,
                markers=markers,
            )
        )
    dropped: list[DroppedSpan] = []
    frame_s = float(1 / Fraction(fps))
    for index in range(draw(st.integers(min_value=0, max_value=3))):
        source = draw(st.sampled_from(sources))
        in_s = draw(st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False))
        extra = draw(
            st.floats(min_value=frame_s * 2, max_value=5, allow_nan=False, allow_infinity=False)
        )
        dropped.append(
            DroppedSpan(source=source.id, in_s=in_s, out_s=in_s + extra, decision=f"d{index:03d}")
        )
    include_words = draw(st.booleans())
    return (
        _sources(fps=fps, width=width, height=height, sources=sources),
        TimelineData(ranges=ranges, dropped=dropped),
        word_starts if include_words else None,
    )


@given(_documents())
@settings(max_examples=40)
def test_time_attributes_are_frame_multiples(document):
    sources, timeline, word_starts = document
    xml = build_fcpxml(sources, timeline, load_profile(), PROJECT, word_starts)
    root = _root(xml)
    frame_dur = _frame_dur(sources.fps)
    for element in root.iter():
        for key, value in element.attrib.items():
            if key not in _TIME_ATTRS:
                continue
            parsed = _parse_time(value)
            assert parsed % frame_dur == 0, value

    fmt = root.find("resources/format")
    assert fmt is not None
    assert fmt.get("colorSpace") == REFERENCE_COLOR_SPACE
    if fmt.get("name") is not None:
        assert fmt.get("name") in _KNOWN_FORMAT_NAMES

    projects = root.findall("library/event/project")
    rough = projects[0].findall("sequence/spine/asset-clip")
    assert len(rough) == len(timeline.ranges)
    offsets = [_parse_time(clip.get("offset")) for clip in rough]
    durations = [_parse_time(clip.get("duration")) for clip in rough]
    assert offsets[0] == 0
    for index in range(len(rough) - 1):
        assert offsets[index + 1] == offsets[index] + durations[index]
    total = sum(durations, Fraction(0))
    sequence = projects[0].find("sequence")
    assert sequence is not None
    assert _parse_time(sequence.get("duration")) == total

    rejects = projects[1].findall("sequence/spine/asset-clip")
    assert len(rejects) == len(timeline.dropped)
    for clip in rejects:
        marker = clip.find("marker")
        assert marker is not None
        assert marker.get("value", "").endswith(": retake")
        assert _parse_time(marker.get("duration")) == frame_dur


def test_dtd_validates_golden(tmp_path: Path):
    if not DTD_PATH.is_file():
        pytest.skip(f"FCPXML DTD not found at {DTD_PATH}")
    xml = _build()
    assert xml.encode("utf-8") == GOLDEN.read_bytes()
    result = export_fcpxml(
        _sources(),
        _timeline(),
        load_profile(),
        PROJECT,
        tmp_path / f"{PROJECT}.fcpxml",
        _word_starts(),
    )
    assert result.dtd_validated is True
    assert result.warning is None
    assert result.path.read_bytes() == GOLDEN.read_bytes()
    dtd = etree.DTD(DTD_PATH)
    assert dtd.validate(etree.parse(result.path)), list(dtd.error_log)
    assert dtd.validate(etree.parse(GOLDEN)), list(dtd.error_log)


@pytest.mark.parametrize("dtd_path", ["", "/no/such/FCPXMLv1_14.dtd"])
def test_missing_dtd_writes_file_and_warns(
    tmp_path: Path, dtd_path: str, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / f"{PROJECT}.fcpxml"
    result = export_fcpxml(
        _sources(),
        _timeline(),
        _profile(dtd_path=dtd_path),
        PROJECT,
        out,
        _word_starts(),
    )
    assert result.dtd_validated is False
    assert result.warning
    assert result.path == out
    assert out.read_bytes() == _build().encode("utf-8")
    assert not (tmp_path / f"{PROJECT}.invalid.fcpxml").exists()
    assert result.warning in capsys.readouterr().err


def test_dtd_failure_writes_invalid_and_raises(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    if not DTD_PATH.is_file():
        pytest.skip(f"FCPXML DTD not found at {DTD_PATH}")
    out = tmp_path / f"{PROJECT}.fcpxml"
    with pytest.raises(FcpxmlValidationError) as raised:
        export_fcpxml(
            _sources(),
            _timeline(),
            _profile(version="9.9"),
            PROJECT,
            out,
            _word_starts(),
        )
    invalid = tmp_path / f"{PROJECT}.invalid.fcpxml"
    assert raised.value.invalid_path == invalid
    assert invalid.is_file()
    assert not out.exists()
    assert raised.value.errors
    captured = capsys.readouterr().err
    assert "DTD" in captured
    assert raised.value.errors[0] in captured


def test_cli_export_writes_the_rough_cut(tmp_path: Path):
    project = _export_project(tmp_path)
    result = CliRunner().invoke(app, ["export", str(project)])
    assert result.exit_code == 0, result.output
    written = project / "out" / f"{PROJECT}.fcpxml"
    assert f"wrote {written}" in result.stdout
    assert written.read_bytes() == GOLDEN.read_bytes()


def test_cli_export_missing_timeline_exits_1(tmp_path: Path):
    result = CliRunner().invoke(app, ["export", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing sources artifact" in result.output


def test_cli_export_dtd_failure_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = _export_project(tmp_path)
    invalid = project / "out" / f"{PROJECT}.invalid.fcpxml"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise FcpxmlValidationError(["bad time"], invalid)

    monkeypatch.setattr("cutter.cli.export_fcpxml", fail)
    result = CliRunner().invoke(app, ["export", str(project)])
    assert result.exit_code == 3
    assert "DTD validation failed" in result.output


def _export_project(tmp_path: Path) -> Path:
    project = tmp_path / PROJECT
    meta = make_meta(
        stage="test",
        stage_version=1,
        inputs_hash="sha256:abc",
        config_hash="sha256:def",
    )
    write_artifact(
        project / "artifacts" / "sources.json",
        SourcesArtifact(meta=meta, data=_sources()),
    )
    write_artifact(
        project / "artifacts" / "timeline.json",
        TimelineArtifact(meta=meta, data=_timeline()),
    )
    write_artifact(
        project / "artifacts" / "words.json",
        WordsArtifact(
            meta=meta,
            data=WordsData(
                words=[
                    Word(
                        i=WORD_INDEX,
                        source="s01",
                        w="hello",
                        norm="hello",
                        start=WORD_START_S,
                        end=WORD_START_S + 0.2,
                        sent=0,
                    )
                ]
            ),
        ),
    )
    return project

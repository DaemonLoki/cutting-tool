"""Write an FCPXML rough cut and rejects project that reference the original files.

The document mirrors ``test/fixtures/reference.fcpxmld/Info.fcpxml``: version
``1.14``, the camera ``<format>`` (including ``name`` and ``colorSpace``), and
the camera ``<asset>`` attributes that still describe our sources. It does not
reproduce that file's screen recordings, titles, color, or transitions.

Camera format::

    <format id="r1" name="FFVideoFormat3840x2160p2398"
            frameDuration="1001/24000s" width="3840" height="2160"
            colorSpace="1-1-1 (Rec. 709)"/>

Camera asset::

    <asset id="r3" name="1-intro" uid="…" start="…" duration="…"
           hasVideo="1" format="r1" hasAudio="1" videoSources="1"
           audioSources="1" audioChannels="2" audioRate="48000">
        <media-rep kind="original-media" sig="…" src="file:///…"/>
    </asset>

``uid``, ``media-rep`` ``sig``, bookmarks, ``<metadata>``, and ``<keyword>``
are Final Cut library data and are not copied. ``audioChannels`` and the
asset ``audioRate`` come from ``SourcesData``. Sequence ``audioRate`` uses
the DTD's kilohertz token (``48k``), as every sequence in the reference does.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from lxml import etree

from cutter.config import Profile
from cutter.models import DroppedSpan, Marker, Range, Source, SourcesData, TimelineData

STAGE_VERSION = 1

# colorSpace on the reference camera format. Kept for every geometry.
REFERENCE_COLOR_SPACE = "1-1-1 (Rec. 709)"

# Named <format> elements in the reference. Match width, height, and the
# rational frame duration. Do not invent names for other geometries.
_REFERENCE_FORMATS: tuple[tuple[int, int, Fraction, str], ...] = (
    (3840, 2160, Fraction("1001/24000"), "FFVideoFormat3840x2160p2398"),
    (7680, 4320, Fraction("100/6000"), "FFVideoFormat7680x4320p60"),
    (3840, 2160, Fraction("1500/90000"), "FFVideoFormat3840x2160p60"),
    (3840, 2160, Fraction("3000/90000"), "FFVideoFormat3840x2160p30"),
)

_FORMAT_ID = "r1"
_REJECT_KIND = "retake"

# Sequence audioRate is an enumeration in FCPXMLv1_14.dtd. Asset audioRate
# stays the integer Hertz from SourcesData, matching the camera asset.
_SEQUENCE_AUDIO_RATE = {
    32000: "32k",
    44100: "44.1k",
    48000: "48k",
    88200: "88.2k",
    96000: "96k",
    176400: "176.4k",
    192000: "192k",
}


class FcpxmlValidationError(Exception):
    """The FCPXML did not validate against the DTD.

    The CLI maps this to exit code 3. The rejected document is at
    ``invalid_path`` (``<project>.invalid.fcpxml`` next to the requested file).
    """

    def __init__(self, errors: list[str], invalid_path: Path) -> None:
        self.errors = errors
        self.invalid_path = invalid_path
        detail = "\n".join(errors)
        super().__init__(f"FCPXML failed DTD validation:\n{detail}")


@dataclass(frozen=True)
class ExportResult:
    """Where the FCPXML was written, and whether a DTD accepted it.

    ``dtd_validated`` is true only when a DTD file was found and the document
    passed. ``warning`` is set when the file was written without validation
    because ``fcpxml.dtd_path`` is empty or the file is missing.
    """

    path: Path
    dtd_validated: bool
    warning: str | None = None


def export_fcpxml(
    sources: SourcesData,
    timeline: TimelineData,
    profile: Profile,
    project: str,
    out_path: Path,
    word_starts: dict[int, float] | None = None,
) -> ExportResult:
    """Build the rough cut and rejects FCPXML and write it to ``out_path``.

    ``word_starts`` maps ``Marker.at_word`` to that word's start in seconds
    from the start of its source file. See ``build_fcpxml``.

    When ``profile.fcpxml.dtd_path`` is empty or not a file, the FCPXML is
    written and ``ExportResult.warning`` explains that validation was skipped.
    When the DTD rejects the document, the file is written to
    ``out/<project>.invalid.fcpxml`` instead, the DTD errors are printed to
    stderr, and ``FcpxmlValidationError`` is raised.
    """
    xml = build_fcpxml(sources, timeline, profile, project, word_starts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warning = _dtd_warning(profile)
    if warning is not None:
        _write(out_path, xml)
        print(warning, file=sys.stderr)
        return ExportResult(path=out_path, dtd_validated=False, warning=warning)

    dtd_path = Path(profile.fcpxml.dtd_path)
    errors = _dtd_errors(xml, dtd_path)
    if errors:
        invalid_path = out_path.parent / f"{project}.invalid.fcpxml"
        _write(invalid_path, xml)
        print("FCPXML DTD validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        raise FcpxmlValidationError(errors, invalid_path)

    _write(out_path, xml)
    return ExportResult(path=out_path, dtd_validated=True, warning=None)


def build_fcpxml(
    sources: SourcesData,
    timeline: TimelineData,
    profile: Profile,
    project: str,
    word_starts: dict[int, float] | None = None,
) -> str:
    """Return the FCPXML document as a UTF-8 XML string.

    ``word_starts`` maps a marker's ``at_word`` index to that word's start
    time in seconds from the start of its source file (the same clock as
    ``Word.start``). The marker is placed at
    ``t(start_frames + frame_of)``, where ``frame_of`` is
    ``floor(word_start * fps)`` using ``fractions.Fraction``, then clamped
    into the asset-clip ``[in_frame, out_frame)``. A marker with no entry
    in the map is placed at the clip's ``in_frame`` (still clamped). Marker
    duration is one frame (``t(1)``).

    Dropped spans store seconds. They become rejects clips with
    ``in_frame = floor(in_s * fps)`` and ``out_frame = ceil(out_s * fps)``.
    Each rejects clip has a marker ``value="<decision id>: retake"``.
    """
    t, fps = _timebase(sources.fps)
    by_id = {source.id: source for source in sources.sources}
    asset_ids = {source.id: f"r{index}" for index, source in enumerate(sources.sources, start=2)}
    audio_layout = _audio_layout(sources.audio_channels)
    sequence_rate = _sequence_audio_rate(sources.audio_rate)

    resources = _element("resources")
    resources.append(_format_element(sources, t))
    for source in sources.sources:
        resources.append(_asset_element(source, asset_ids[source.id], sources, t))

    rough_spine, rough_frames = _rough_cut_spine(
        timeline.ranges, by_id, asset_ids, t, fps, profile.fcpxml.audio_role, word_starts
    )
    rejects_spine, reject_frames = _rejects_spine(
        timeline.dropped, by_id, asset_ids, t, fps, profile
    )

    event = _element("event", {"name": _fill(profile.fcpxml.event_name, project)})
    event.append(
        _project(
            _fill(profile.fcpxml.project_name, project),
            rough_spine,
            t(rough_frames),
            t(0),
            audio_layout,
            sequence_rate,
        )
    )
    event.append(
        _project(
            _fill(profile.fcpxml.rejects_project_name, project),
            rejects_spine,
            t(reject_frames),
            t(0),
            audio_layout,
            sequence_rate,
        )
    )
    library = _element("library", children=[event])
    root = _element(
        "fcpxml",
        {"version": profile.fcpxml.version},
        [resources, library],
    )
    return _serialize(root)


def _timebase(fps: str) -> tuple[Callable[[int], str], Fraction]:
    """Return ``t(frames)`` and the fps fraction.

    ``frame_dur`` is ``denominator/numerator`` of ``fps`` (``1001/24000`` when
    fps is ``24000/1001``). The division stays in ``Fraction``; ``int / int``
    would be a float.
    """
    rate = Fraction(fps)
    if rate <= 0:
        raise ValueError(f"fps must be a positive fraction, got {fps!r}")
    frame_dur = Fraction(rate.denominator, rate.numerator)

    def t(frames: int) -> str:
        value = Fraction(frames) * frame_dur
        if value == 0:
            return "0s"
        return f"{value.numerator}/{value.denominator}s"

    return t, rate


def _format_element(sources: SourcesData, t: Callable[[int], str]) -> etree._Element:
    frame_dur = Fraction(sources.fps)
    frame_dur = Fraction(frame_dur.denominator, frame_dur.numerator)
    attrs: dict[str, str] = {"id": _FORMAT_ID}
    name = _reference_format_name(sources.width, sources.height, frame_dur)
    if name is not None:
        attrs["name"] = name
    attrs["frameDuration"] = t(1)
    attrs["width"] = str(sources.width)
    attrs["height"] = str(sources.height)
    attrs["colorSpace"] = REFERENCE_COLOR_SPACE
    return _element("format", attrs)


def _reference_format_name(width: int, height: int, frame_dur: Fraction) -> str | None:
    for known_width, known_height, known_dur, name in _REFERENCE_FORMATS:
        if width == known_width and height == known_height and frame_dur == known_dur:
            return name
    return None


def _asset_element(
    source: Source,
    asset_id: str,
    sources: SourcesData,
    t: Callable[[int], str],
) -> etree._Element:
    # Attribute order follows the reference camera asset, without uid.
    attrs = {
        "id": asset_id,
        "name": Path(source.path).stem,
        "start": t(source.start_frames),
        "duration": t(source.duration_frames),
        "hasVideo": "1",
        "format": _FORMAT_ID,
        "hasAudio": "1",
        "videoSources": "1",
        "audioSources": "1",
        "audioChannels": str(sources.audio_channels),
        "audioRate": str(sources.audio_rate),
    }
    media = _element(
        "media-rep",
        {"kind": "original-media", "src": Path(source.path).resolve().as_uri()},
    )
    return _element("asset", attrs, [media])


def _rough_cut_spine(
    ranges: list[Range],
    by_id: dict[str, Source],
    asset_ids: dict[str, str],
    t: Callable[[int], str],
    fps: Fraction,
    audio_role: str,
    word_starts: dict[int, float] | None,
) -> tuple[etree._Element, int]:
    clips: list[etree._Element] = []
    offset = 0
    for item in ranges:
        source = _lookup(by_id, item.source, f"range {item.id}")
        duration = item.out_frame - item.in_frame
        if duration < 0:
            raise ValueError(f"range {item.id} has out_frame before in_frame")
        markers = [
            _marker_element(
                t(source.start_frames + _marker_frame(marker, item, fps, word_starts)),
                t(1),
                marker.text,
            )
            for marker in item.markers
        ]
        clips.append(
            _asset_clip(
                ref=asset_ids[source.id],
                name=Path(source.path).stem,
                offset=t(offset),
                start=t(source.start_frames + item.in_frame),
                duration=t(duration),
                audio_role=audio_role,
                children=markers,
            )
        )
        offset += duration
    return _element("spine", children=clips), offset


def _rejects_spine(
    dropped: list[DroppedSpan],
    by_id: dict[str, Source],
    asset_ids: dict[str, str],
    t: Callable[[int], str],
    fps: Fraction,
    profile: Profile,
) -> tuple[etree._Element, int]:
    clips: list[etree._Element] = []
    offset = 0
    for span in dropped:
        source = _lookup(by_id, span.source, f"dropped span {span.decision}")
        in_frame = math.floor(Fraction(span.in_s) * fps)
        out_frame = math.ceil(Fraction(span.out_s) * fps)
        duration = out_frame - in_frame
        if duration < 0:
            raise ValueError(f"dropped span {span.decision} ends before it starts")
        marker = _marker_element(
            t(source.start_frames + in_frame),
            t(1),
            f"{span.decision}: {_REJECT_KIND}",
        )
        clips.append(
            _asset_clip(
                ref=asset_ids[source.id],
                name=Path(source.path).stem,
                offset=t(offset),
                start=t(source.start_frames + in_frame),
                duration=t(duration),
                audio_role=profile.fcpxml.audio_role,
                children=[marker],
            )
        )
        offset += duration
    return _element("spine", children=clips), offset


def _marker_frame(
    marker: Marker,
    item: Range,
    fps: Fraction,
    word_starts: dict[int, float] | None,
) -> int:
    if word_starts is not None and marker.at_word in word_starts:
        frame = math.floor(Fraction(word_starts[marker.at_word]) * fps)
    else:
        frame = item.in_frame
    last = item.out_frame - 1
    if last < item.in_frame:
        return item.in_frame
    return min(max(frame, item.in_frame), last)


def _asset_clip(
    *,
    ref: str,
    name: str,
    offset: str,
    start: str,
    duration: str,
    audio_role: str,
    children: list[etree._Element],
) -> etree._Element:
    # Camera asset-clips in the reference omit format; they inherit the sequence.
    return _element(
        "asset-clip",
        {
            "ref": ref,
            "offset": offset,
            "name": name,
            "start": start,
            "duration": duration,
            "tcFormat": "NDF",
            "audioRole": audio_role,
        },
        children,
    )


def _marker_element(start: str, duration: str, value: str) -> etree._Element:
    return _element("marker", {"start": start, "duration": duration, "value": value})


def _project(
    name: str,
    spine: etree._Element,
    duration: str,
    tc_start: str,
    audio_layout: str,
    audio_rate: str,
) -> etree._Element:
    sequence = _element(
        "sequence",
        {
            "format": _FORMAT_ID,
            "duration": duration,
            "tcStart": tc_start,
            "tcFormat": "NDF",
            "audioLayout": audio_layout,
            "audioRate": audio_rate,
        },
        [spine],
    )
    return _element("project", {"name": name}, [sequence])


def _audio_layout(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    if channels > 2:
        return "surround"
    raise ValueError(f"audio_channels must be positive, got {channels}")


def _sequence_audio_rate(audio_rate: int) -> str:
    try:
        return _SEQUENCE_AUDIO_RATE[audio_rate]
    except KeyError as exc:
        raise ValueError(f"audio_rate {audio_rate} has no FCPXML sequence rate") from exc


def _lookup(by_id: dict[str, Source], source_id: str, label: str) -> Source:
    try:
        return by_id[source_id]
    except KeyError as exc:
        raise ValueError(f"{label} references unknown source {source_id!r}") from exc


def _fill(template: str, project: str) -> str:
    return template.replace("{project}", project)


def _element(
    tag: str,
    attrs: dict[str, str] | None = None,
    children: list[etree._Element] | None = None,
) -> etree._Element:
    element = etree.Element(tag)
    for key, value in (attrs or {}).items():
        element.set(key, value)
    for child in children or []:
        element.append(child)
    return element


def _serialize(root: etree._Element) -> str:
    raw = etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        doctype="<!DOCTYPE fcpxml>",
    )
    text = raw.decode("utf-8")
    declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    single = "<?xml version='1.0' encoding='UTF-8'?>"
    if text.startswith(single):
        text = declaration + text[len(single) :]
    if not text.endswith("\n"):
        text += "\n"
    return text


def _write(path: Path, xml: str) -> None:
    path.write_bytes(xml.encode("utf-8"))


def _dtd_warning(profile: Profile) -> str | None:
    raw = profile.fcpxml.dtd_path.strip()
    if not raw:
        return "warning: fcpxml.dtd_path is not set; wrote FCPXML without DTD validation"
    path = Path(raw)
    if not path.is_file():
        return f"warning: FCPXML DTD not found at {path}; wrote FCPXML without DTD validation"
    return None


def _dtd_errors(xml: str, dtd_path: Path) -> list[str]:
    dtd = etree.DTD(dtd_path)
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    document = etree.fromstring(xml.encode("utf-8"), parser)
    if dtd.validate(document):
        return []
    return [str(entry) for entry in dtd.error_log]

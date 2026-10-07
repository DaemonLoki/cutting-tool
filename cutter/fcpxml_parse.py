"""Parse a Final Cut primary storyline into file-relative source spans.

A gold edit's spine is the sequence of ranges a person kept. Each span is
seconds from the start of one source. Gaps, titles, generators, and connected
clips are not ranges. Unsupported spine children are skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree

logger = logging.getLogger("cutter.fcpxml_parse")

_IGNORED_SPINE_TAGS = frozenset({"gap", "title", "generator"})
_FRACTION_TIME = re.compile(r"(\d+)/(\d+)")
_DECIMAL_TIME = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class SourceSpan:
    """A kept span on one source, in seconds from the start of the file."""

    source_name: str
    in_s: float
    out_s: float


@dataclass(frozen=True)
class ParseResult:
    """Kept source spans, plus unsupported spine children in document order."""

    spans: list[SourceSpan]
    skipped: list[str]


@dataclass(frozen=True)
class _Asset:
    source_name: str
    start: Fraction


def parse_fcpxml(path: Path) -> ParseResult:
    """Read the first project's sequence spine as file-relative source spans.

    ``asset-clip`` elements, and ``clip`` elements with a nested ``video`` or
    ``audio`` ref, become spans on that source. ``gap``, ``title``, and
    ``generator`` are ignored. ``sync-clip``, ``mc-clip``, ``ref-clip``, and
    any other direct spine child are skipped and logged once.

    Raises FileNotFoundError when ``path`` is missing, and ValueError when
    the document has no project or no spine.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.parse(str(path), parser).getroot()
    assets = _assets(root)
    spine = _first_spine(root, path)
    spans: list[SourceSpan] = []
    skipped: list[str] = []
    for child in spine:
        tag = _local(child)
        if not tag or tag in _IGNORED_SPINE_TAGS:
            continue
        if tag == "asset-clip":
            spans.append(_span_from_asset_clip(child, assets))
        elif tag == "clip":
            span = _span_from_clip(child, assets)
            if span is None:
                skipped.append(tag)
            else:
                spans.append(span)
        else:
            skipped.append(tag)

    if skipped:
        logger.warning("skipped unsupported spine children: %s", ", ".join(skipped))
    return ParseResult(spans=spans, skipped=skipped)


def _assets(root: etree._Element) -> dict[str, _Asset]:
    found: dict[str, _Asset] = {}
    for element in root.iter():
        if _local(element) != "asset":
            continue
        asset_id = element.get("id")
        src = _original_src(element)
        if not asset_id or src is None:
            continue
        found[asset_id] = _Asset(_source_name(src), _parse_time(element.get("start") or "0s"))
    return found


def _original_src(asset: etree._Element) -> str | None:
    for element in asset.iter():
        if _local(element) == "media-rep" and element.get("kind") == "original-media":
            return element.get("src")
    return None


def _source_name(src: str) -> str:
    return Path(unquote(urlparse(src).path)).name


def _first_spine(root: etree._Element, path: Path) -> etree._Element:
    project = next((element for element in root.iter() if _local(element) == "project"), None)
    if project is None:
        raise ValueError(f"{path} has no project")
    for element in project.iter():
        if element is project or _local(element) != "sequence":
            continue
        for child in element:
            if _local(child) == "spine":
                return child
    raise ValueError(f"{path} has no spine")


def _span_from_asset_clip(element: etree._Element, assets: dict[str, _Asset]) -> SourceSpan:
    ref = element.get("ref")
    asset = _lookup_asset(ref, assets, "asset-clip")
    duration = element.get("duration")
    if duration is None:
        raise ValueError(f"asset-clip ref {ref!r} is missing duration")
    return _make_span(asset, _parse_time(element.get("start") or "0s"), _parse_time(duration))


def _span_from_clip(element: etree._Element, assets: dict[str, _Asset]) -> SourceSpan | None:
    media = _nested_av(element)
    if media is None:
        return None
    ref = media.get("ref")
    asset = _lookup_asset(ref, assets, "clip")
    start = element.get("start") or media.get("start") or "0s"
    duration = element.get("duration") or media.get("duration")
    if duration is None:
        raise ValueError(f"clip ref {ref!r} is missing duration")
    return _make_span(asset, _parse_time(start), _parse_time(duration))


def _nested_av(clip: etree._Element) -> etree._Element | None:
    for element in clip.iter():
        if element is clip:
            continue
        if _local(element) in {"video", "audio"} and element.get("ref"):
            return element
    return None


def _lookup_asset(ref: str | None, assets: dict[str, _Asset], label: str) -> _Asset:
    if ref is None or ref not in assets:
        raise ValueError(f"{label} ref {ref!r} does not match an asset")
    return assets[ref]


def _make_span(asset: _Asset, start: Fraction, duration: Fraction) -> SourceSpan:
    origin = start - asset.start
    return SourceSpan(asset.source_name, float(origin), float(origin + duration))


def _parse_time(token: str) -> Fraction:
    """Parse ``0s``, ``12.5s``, or ``1001/24000s`` as rational seconds."""
    if token.endswith("s"):
        body = token[:-1]
        fraction = _FRACTION_TIME.fullmatch(body)
        if fraction is not None:
            denominator = int(fraction.group(2))
            if denominator != 0:
                return Fraction(int(fraction.group(1)), denominator)
        elif _DECIMAL_TIME.fullmatch(body) is not None:
            return Fraction(body)
    raise ValueError(f"invalid FCPXML time {token!r}")


def _local(element: etree._Element) -> str:
    tag = element.tag
    if not isinstance(tag, str):
        return ""
    return tag.rpartition("}")[2]

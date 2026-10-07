"""Pick ``long`` or ``short`` from the frame when ``--profile`` is omitted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cutter.config import load_profile
from cutter.ingest import _require_tools, list_sources, probe_file
from cutter.models import SourcesArtifact


class FrameUnknown(Exception):
    """The project has no raw sources and no sources artifact to measure."""


@dataclass(frozen=True)
class ProfileChoice:
    """The profile selected from a frame, plus that frame's display size."""

    name: str
    width: int
    height: int

    @property
    def orientation(self) -> str:
        if self.height > self.width:
            return "vertical"
        if self.width > self.height:
            return "horizontal"
        return "square"


def profile_name_for_frame(width: int, height: int) -> str:
    """``short`` when the displayed frame is taller than it is wide.

    Horizontal and square frames use ``long``.
    """
    if height > width:
        return "short"
    return "long"


def oriented_size(width: int, height: int, rotation: int = 0) -> tuple[int, int]:
    """Width and height after a 90° or 270° display rotation."""
    if rotation % 180 == 90:
        return height, width
    return width, height


def choose_profile(project_dir: Path) -> ProfileChoice:
    """Read the first source, or ``sources.json`` when ``raw/`` is absent.

    A phone file stored sideways is measured after its rotation tag, so a
    portrait picture selects ``short`` even when the stored pixels are
    landscape.
    """
    project_dir = Path(project_dir)
    raw = project_dir / "raw"
    if raw.is_dir():
        _require_tools()
        info = probe_file(list_sources(raw, _extensions())[0])
        width, height = oriented_size(info.width, info.height, info.rotation)
        return ProfileChoice(profile_name_for_frame(width, height), width, height)
    artifact = project_dir / "artifacts" / "sources.json"
    if artifact.is_file():
        data = SourcesArtifact.model_validate_json(artifact.read_text(encoding="utf-8")).data
        return ProfileChoice(
            profile_name_for_frame(data.width, data.height),
            data.width,
            data.height,
        )
    raise FrameUnknown(f"no raw folder at {raw}")


def _extensions() -> list[str]:
    found: list[str] = []
    for name in ("long", "short"):
        for ext in load_profile(name).ingest.allowed_extensions:
            if ext not in found:
                found.append(ext)
    return found

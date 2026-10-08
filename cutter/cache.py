"""Stage cache: hash inputs and profile sections, then skip unchanged work."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from cutter.config import Profile
from cutter.models import ArtifactMeta

# Sections of the profile each stage reads. A change outside that section
# leaves the stage's config hash alone.
STAGE_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "ingest": ("ingest",),
    "transcribe": ("transcribe",),
    "audio": ("vad", "claps"),
    "align": ("script", "chapters"),
    "retakes": ("retakes", "claps", "script"),
    "fillers": ("fillers",),
    "judge": ("judge",),
    "tighten": ("tighten", "vad", "claps", "chapters"),
    "export": ("fcpxml",),
}


def inputs_hash(
    *,
    artifacts: Sequence[Path] = (),
    media: Sequence[Path] = (),
) -> str:
    """Hash artifact bytes, and size plus mtime for raw media.

    Media contents are not read. File names must be unique within each group.
    """
    digest = hashlib.sha256()
    for path in _by_name(artifacts, "artifact"):
        digest.update(b"\0artifact\0")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_file_digest(path).encode())
    for path in _by_name(media, "media"):
        stat = path.stat()
        digest.update(b"\0media\0")
        digest.update(path.name.encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return f"sha256:{digest.hexdigest()}"


def config_hash(profile: Profile, sections: Sequence[str]) -> str:
    """Hash only the profile sections a stage reads."""
    data = profile.model_dump(mode="json")
    missing = [section for section in sections if section not in data]
    if missing:
        raise ValueError(f"unknown config sections: {', '.join(missing)}")
    subset = {section: data[section] for section in sections}
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def cache_hit(
    path: Path,
    *,
    stage: str,
    stage_version: int,
    inputs_hash: str,
    config_hash: str,
) -> bool:
    """True when the artifact on disk was built from these inputs and this config.

    Each stage module declares its own ``STAGE_VERSION``. Bumping it misses.
    A missing or unreadable artifact is a miss.
    """
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = ArtifactMeta.model_validate(payload["meta"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return False
    return (
        meta.stage == stage
        and meta.stage_version == stage_version
        and meta.inputs_hash == inputs_hash
        and meta.config_hash == config_hash
    )


def _by_name(paths: Sequence[Path], label: str) -> list[Path]:
    ordered = sorted(paths, key=lambda path: path.name)
    names = [path.name for path in ordered]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} paths must have unique file names")
    return ordered


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

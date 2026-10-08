"""Phase 2 skeleton for filler removal.

T4 fills in filler decisions. Until then this stage writes an empty
``fillers.json`` so the pipeline can cache the stage before it drops words.
"""

from __future__ import annotations

from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, load_profile
from cutter.models import DecisionsArtifact, DecisionsData, make_meta, write_artifact

STAGE = "fillers"
STAGE_VERSION = 1


def run_fillers(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> DecisionsArtifact:
    """Stage entry for ``cutter fillers <project_dir> [--profile] [--force]``.

    Reads ``artifacts/words.json`` and ``artifacts/decisions.json``. Writes
    ``artifacts/fillers.json`` with no decisions. A cache hit returns the
    artifact already on disk.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    fillers_path = artifacts / "fillers.json"
    for label, path in (("words", words_path), ("decisions", decisions_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} artifact: {path}")

    hashed_inputs = inputs_hash(artifacts=[words_path, decisions_path])
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        fillers_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return DecisionsArtifact.model_validate_json(fillers_path.read_text(encoding="utf-8"))

    artifact = DecisionsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=DecisionsData(decisions=[]),
    )
    write_artifact(fillers_path, artifact)
    return artifact

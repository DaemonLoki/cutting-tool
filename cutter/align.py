"""Phase 2 skeleton for script alignment.

T5 fills in script parsing and take choice. Until then this stage writes an
empty ``alignment.json`` so a missing script and a present one both cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, load_profile
from cutter.models import AlignmentArtifact, AlignmentData, make_meta, write_artifact

STAGE = "align"
STAGE_VERSION = 1

logger = logging.getLogger("cutter.align")


def run_align(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> AlignmentArtifact:
    """Stage entry for ``cutter align <project_dir> [--profile] [--force]``.

    Reads ``artifacts/words.json`` and ``<project>/<script.path>`` when that
    file exists. Writes ``artifacts/alignment.json``. The skeleton always
    stores ``script: null`` and empty lists. A cache hit returns the artifact
    already on disk.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    alignment_path = artifacts / "alignment.json"
    if not words_path.is_file():
        raise FileNotFoundError(f"missing words artifact: {words_path}")

    script_path = project_dir / loaded.script.path
    hashed = [words_path]
    if script_path.is_file():
        hashed.append(script_path)
    hashed_inputs = inputs_hash(artifacts=hashed)
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        alignment_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return AlignmentArtifact.model_validate_json(alignment_path.read_text(encoding="utf-8"))

    if not script_path.is_file():
        logger.info("no script at %s", script_path)
    artifact = AlignmentArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=AlignmentData(
            script=None,
            chapters=[],
            sentences=[],
            unscripted=[],
            missing=[],
        ),
    )
    write_artifact(alignment_path, artifact)
    return artifact

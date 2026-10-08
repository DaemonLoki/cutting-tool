"""Phase 2 skeleton for speech segments and claps.

T2 fills in voice activity and T3 fills in clap detection. Until then this
stage writes an empty ``audio_events.json`` so later stages can cache against it.
"""

from __future__ import annotations

from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, load_profile
from cutter.models import (
    AudioEventsArtifact,
    AudioEventsData,
    Source,
    SourcesArtifact,
    make_meta,
    write_artifact,
)

STAGE = "audio"
STAGE_VERSION = 1


def run_audio(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> AudioEventsArtifact:
    """Stage entry for ``cutter audio <project_dir> [--profile] [--force]``.

    Reads ``artifacts/sources.json`` and each source's 16 kHz and 48 kHz WAVs.
    Writes ``artifacts/audio_events.json``. A cache hit returns the artifact
    already on disk. The skeleton records ``backend: none`` and no events.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    sources_path = artifacts / "sources.json"
    events_path = artifacts / "audio_events.json"
    if not sources_path.is_file():
        raise FileNotFoundError(f"missing sources artifact: {sources_path}")

    sources = SourcesArtifact.model_validate_json(sources_path.read_text(encoding="utf-8"))
    wavs = _wavs(project_dir, sources.data.sources)
    hashed_inputs = inputs_hash(artifacts=[sources_path, *wavs])
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        events_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return AudioEventsArtifact.model_validate_json(events_path.read_text(encoding="utf-8"))

    artifact = AudioEventsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=AudioEventsData(backend="none", speech=[], claps=[]),
    )
    write_artifact(events_path, artifact)
    return artifact


def _wavs(project_dir: Path, sources: list[Source]) -> list[Path]:
    wavs: list[Path] = []
    for source in sources:
        asr = project_dir / source.asr_wav
        analysis = project_dir / source.analysis_wav
        if not asr.is_file():
            raise FileNotFoundError(f"missing asr wav for {source.id}: {asr}")
        if not analysis.is_file():
            raise FileNotFoundError(f"missing analysis audio: {analysis}")
        wavs.extend((asr, analysis))
    return wavs

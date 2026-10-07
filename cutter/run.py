"""Run every stage in order and skip work the cache already covers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile
from cutter.fcpxml import STAGE_VERSION as EXPORT_STAGE_VERSION
from cutter.fcpxml import ExportResult, export_fcpxml
from cutter.ingest import ingest_project
from cutter.judge import judge_cached, run_judge
from cutter.models import (
    Artifact,
    DecisionsArtifact,
    SourcesData,
    StrictModel,
    TimelineData,
    Word,
    make_meta,
    write_artifact,
)
from cutter.retakes import run_retakes
from cutter.tighten import run_tighten
from cutter.transcribe import transcribe_project

_EXPORT_STAGE = "export"


class _ExportData(StrictModel):
    dtd_validated: bool
    warning: str | None = None


class _ExportArtifact(Artifact[_ExportData]):
    data: _ExportData


@dataclass(frozen=True)
class RunSummary:
    """What ``cutter run`` prints after the rough cut is written."""

    sources: int
    raw_duration_s: float
    output_duration_s: float
    dropped: int
    kept: int
    flagged: int
    fcpxml: Path


def run_project(
    project_dir: Path,
    profile: Profile,
    *,
    no_llm: bool = False,
    force: bool = False,
) -> RunSummary:
    """Ingest, transcribe, retake, judge, tighten, and export.

    A judged ``decisions.json`` already covers retakes, so a cache hit skips
    both. Running retakes again would replace that file and send the same
    takes back to the model.
    """
    project_dir = Path(project_dir)
    sources = ingest_project(project_dir, profile, force=force)
    words = transcribe_project(project_dir, profile=profile, force=force)
    decisions = _decisions(project_dir, profile, no_llm=no_llm, force=force)
    timeline = run_tighten(project_dir, profile, force=force)
    exported = _export(
        project_dir,
        profile,
        sources,
        timeline.data,
        words.data.words,
        force=force,
    )
    choices = decisions.data.decisions
    return RunSummary(
        sources=len(sources.sources),
        raw_duration_s=sum(source.duration_s for source in sources.sources),
        output_duration_s=sum(item.out_s - item.in_s for item in timeline.data.ranges),
        dropped=sum(decision.action == "drop" for decision in choices),
        kept=sum(decision.action == "keep" for decision in choices),
        flagged=sum(decision.flag for decision in choices),
        fcpxml=exported.path,
    )


def _decisions(
    project_dir: Path,
    profile: Profile,
    *,
    no_llm: bool,
    force: bool,
) -> DecisionsArtifact:
    decisions_path = project_dir / "artifacts" / "decisions.json"
    if force or not judge_cached(project_dir, profile, no_llm=no_llm):
        run_retakes(project_dir, profile, force=force)
        return run_judge(project_dir, profile, force=force, no_llm=no_llm)
    return DecisionsArtifact.model_validate_json(decisions_path.read_text(encoding="utf-8"))


def _export(
    project_dir: Path,
    profile: Profile,
    sources: SourcesData,
    timeline: TimelineData,
    words: list[Word],
    *,
    force: bool,
) -> ExportResult:
    artifacts = project_dir / "artifacts"
    out_path = project_dir / "out" / f"{project_dir.name}.fcpxml"
    stamp_path = artifacts / "export.json"
    hashed_inputs = _export_inputs(artifacts)
    hashed_config = config_hash(profile, STAGE_CONFIG_SECTIONS[_EXPORT_STAGE])
    if (
        not force
        and out_path.is_file()
        and cache_hit(
            stamp_path,
            stage=_EXPORT_STAGE,
            stage_version=EXPORT_STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        )
    ):
        stamped = _ExportArtifact.model_validate_json(stamp_path.read_text(encoding="utf-8"))
        return ExportResult(
            path=out_path,
            dtd_validated=stamped.data.dtd_validated,
            warning=stamped.data.warning,
        )

    result = export_fcpxml(
        sources,
        timeline,
        profile,
        project_dir.name,
        out_path,
        {word.i: word.start for word in words},
    )
    write_artifact(
        stamp_path,
        _ExportArtifact(
            meta=make_meta(
                stage=_EXPORT_STAGE,
                stage_version=EXPORT_STAGE_VERSION,
                inputs_hash=hashed_inputs,
                config_hash=hashed_config,
            ),
            data=_ExportData(dtd_validated=result.dtd_validated, warning=result.warning),
        ),
    )
    return result


def _export_inputs(artifacts: Path) -> str:
    paths = [artifacts / "sources.json", artifacts / "timeline.json"]
    words = artifacts / "words.json"
    if words.is_file():
        paths.append(words)
    return inputs_hash(artifacts=paths)

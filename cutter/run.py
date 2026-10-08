"""Run every stage in order and skip work the cache already covers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cutter.align import run_align
from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile
from cutter.events import run_audio
from cutter.fcpxml import STAGE_VERSION as EXPORT_STAGE_VERSION
from cutter.fcpxml import ExportResult, export_fcpxml
from cutter.fillers import run_fillers
from cutter.ingest import ingest_project
from cutter.judge import judge_cached, run_judge
from cutter.models import (
    AlignmentArtifact,
    Artifact,
    AudioEventsArtifact,
    DecisionsArtifact,
    SourcesData,
    StrictModel,
    TimelineArtifact,
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
    """What ``cutter run`` prints after the rough cut is written.

    ``speech_share`` is speech-segment duration divided by raw duration, or 0
    when there is no speech. The other Phase 2 counts are read from the
    artifacts this run wrote. A missing optional artifact counts as zero.
    """

    sources: int
    raw_duration_s: float
    output_duration_s: float
    dropped: int
    kept: int
    flagged: int
    fcpxml: Path
    speech_share: float
    claps: int
    filler_drops: int
    script_sentences_found: int
    script_sentences_missing: int
    unscripted_spans: int
    chapters: int


def run_project(
    project_dir: Path,
    profile: Profile,
    *,
    no_llm: bool = False,
    force: bool = False,
) -> RunSummary:
    """Ingest, transcribe, then the Phase 2 scaffolds, retakes, judge, tighten, and export.

    A judged ``decisions.json`` already covers retakes, so a cache hit skips
    both. Running retakes again would replace that file and send the same
    takes back to the model. Fillers hashes that file, so it runs only after
    the judged artifact is the one on disk. The spec lists fillers before
    judge; doing that would miss the filler cache on the next run, because
    judge rewrites ``decisions.json``.
    """
    project_dir = Path(project_dir)
    sources = ingest_project(project_dir, profile, force=force)
    words = transcribe_project(project_dir, profile=profile, force=force)
    run_audio(project_dir, profile, force=force)
    run_align(project_dir, profile, force=force)
    decisions = _decisions(project_dir, profile, no_llm=no_llm, force=force)
    run_fillers(project_dir, profile, force=force)
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
    raw_duration_s = sum(source.duration_s for source in sources.sources)
    counts = _phase2_counts(project_dir, raw_duration_s)
    return RunSummary(
        sources=len(sources.sources),
        raw_duration_s=raw_duration_s,
        output_duration_s=sum(item.out_s - item.in_s for item in timeline.data.ranges),
        dropped=sum(decision.action == "drop" for decision in choices),
        kept=sum(decision.action == "keep" for decision in choices),
        flagged=sum(decision.flag for decision in choices),
        fcpxml=exported.path,
        speech_share=counts.speech_share,
        claps=counts.claps,
        filler_drops=counts.filler_drops,
        script_sentences_found=counts.script_sentences_found,
        script_sentences_missing=counts.script_sentences_missing,
        unscripted_spans=counts.unscripted_spans,
        chapters=counts.chapters,
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


@dataclass(frozen=True)
class _Phase2Counts:
    speech_share: float
    claps: int
    filler_drops: int
    script_sentences_found: int
    script_sentences_missing: int
    unscripted_spans: int
    chapters: int


def _phase2_counts(project_dir: Path, raw_duration_s: float) -> _Phase2Counts:
    """Read the Phase 2 summary from artifacts already on disk.

    Speech share is the total speech-segment duration divided by raw duration.
    It is 0 when there is no speech. Claps come from ``audio_events.json``.
    Filler drops are filler decisions whose action is drop. Script sentences
    found are sentences that have a take; missing and unscripted spans come
    from ``alignment.json``. Chapters are the markers on ``timeline.json``.
    Each of those files is optional here: a missing file counts as zero.
    """
    artifacts = project_dir / "artifacts"
    speech_s = 0.0
    claps = 0
    audio_path = artifacts / "audio_events.json"
    if audio_path.is_file():
        audio = AudioEventsArtifact.model_validate_json(audio_path.read_text(encoding="utf-8"))
        speech_s = sum(segment.end - segment.start for segment in audio.data.speech)
        claps = len(audio.data.claps)
    if speech_s == 0.0 or raw_duration_s == 0.0:
        speech_share = 0.0
    else:
        speech_share = speech_s / raw_duration_s

    filler_drops = 0
    fillers_path = artifacts / "fillers.json"
    if fillers_path.is_file():
        fillers = DecisionsArtifact.model_validate_json(fillers_path.read_text(encoding="utf-8"))
        filler_drops = sum(decision.action == "drop" for decision in fillers.data.decisions)

    found = 0
    missing = 0
    unscripted = 0
    alignment_path = artifacts / "alignment.json"
    if alignment_path.is_file():
        alignment = AlignmentArtifact.model_validate_json(
            alignment_path.read_text(encoding="utf-8")
        )
        found = sum(1 for sentence in alignment.data.sentences if sentence.takes)
        missing = len(alignment.data.missing)
        unscripted = len(alignment.data.unscripted)

    chapters = 0
    timeline_path = artifacts / "timeline.json"
    if timeline_path.is_file():
        timeline = TimelineArtifact.model_validate_json(timeline_path.read_text(encoding="utf-8"))
        chapters = len(timeline.data.chapters)

    return _Phase2Counts(
        speech_share=speech_share,
        claps=claps,
        filler_drops=filler_drops,
        script_sentences_found=found,
        script_sentences_missing=missing,
        unscripted_spans=unscripted,
        chapters=chapters,
    )

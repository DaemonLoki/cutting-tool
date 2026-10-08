"""Cutter commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from cutter.align import run_align
from cutter.choose import FrameUnknown, choose_profile
from cutter.config import REPO_ROOT, ConfigError, Profile, load_profile
from cutter.evaluate import evaluate_gold
from cutter.events import run_audio
from cutter.fcpxml import FcpxmlValidationError, export_fcpxml
from cutter.fillers import run_fillers
from cutter.ingest import IngestError, ToolMissing, ingest_project
from cutter.judge import run_judge
from cutter.models import (
    AlignmentArtifact,
    AudioEventsArtifact,
    Decision,
    SourcesArtifact,
    TimelineArtifact,
    WordsArtifact,
)
from cutter.retakes import run_retakes
from cutter.run import RunSummary, run_project
from cutter.tighten import run_tighten
from cutter.transcribe import TimestampError, transcribe_project, words_between

app = typer.Typer(
    help="Make an editable rough cut from one project's English sources.",
    no_args_is_help=True,
)

ProfileOpt = Annotated[
    str | None,
    typer.Option(
        "--profile",
        help=(
            "Profile under profiles/. Omit to use long for a horizontal or "
            "square frame and short for a vertical one."
        ),
    ),
]
ForceOpt = Annotated[bool, typer.Option("--force", help="Re-run even when the cache matches.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override one profile value: key.path=value. Repeatable."),
]


def main() -> None:
    app()


@app.command()
def ingest(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    profile: ProfileOpt = None,
    force: ForceOpt = False,
) -> None:
    """Validate sources and extract audio."""
    try:
        loaded = _resolve_profile(project_dir, profile)
        data = ingest_project(project_dir, loaded, force=force)
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ToolMissing as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    artifact = project_dir / "artifacts" / "sources.json"
    typer.echo(f"ingested {len(data.sources)} sources -> {artifact}")


@app.command()
def transcribe(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
) -> None:
    """Transcribe speech to words."""
    try:
        loaded = _resolve_profile(project_dir)
        artifact = transcribe_project(project_dir, profile=loaded, force=force)
    except (FileNotFoundError, TimestampError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    words_path = project_dir / "artifacts" / "words.json"
    typer.echo(f"transcribed {len(artifact.data.words)} words -> {words_path}")


@app.command()
def audio(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    profile: ProfileOpt = None,
    force: ForceOpt = False,
) -> None:
    """Record speech segments and claps.

    Speech comes from the profile VAD backend. Claps are not detected yet.
    """
    sources_path = project_dir / "artifacts" / "sources.json"
    if not sources_path.is_file():
        typer.echo(f"missing sources artifact: {sources_path}", err=True)
        raise typer.Exit(1)
    try:
        loaded = _resolve_profile(project_dir, profile)
        artifact = run_audio(project_dir, loaded, force=force)
    except (FileNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _echo_audio(artifact)


@app.command()
def align(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    profile: ProfileOpt = None,
    force: ForceOpt = False,
) -> None:
    """Align the transcript to an optional script.

    This stage is a scaffold. It writes an empty artifact.
    """
    _configure_logging()
    words_path = project_dir / "artifacts" / "words.json"
    if not words_path.is_file():
        typer.echo(f"missing words artifact: {words_path}", err=True)
        raise typer.Exit(1)
    try:
        loaded = _resolve_profile(project_dir, profile)
        artifact = run_align(project_dir, loaded, force=force)
    except (FileNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _echo_alignment(artifact)


@app.command()
def retakes(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
) -> None:
    """Find aborted takes and decide which to drop."""
    words_path = project_dir / "artifacts" / "words.json"
    if not words_path.is_file():
        typer.echo(f"missing words artifact: {words_path}", err=True)
        raise typer.Exit(1)
    try:
        loaded = _resolve_profile(project_dir)
        artifact = run_retakes(project_dir, loaded, force=force)
    except (FileNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _echo_decisions(artifact.data.decisions, project_dir / "artifacts" / "decisions.json")


@app.command()
def judge(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Leave flagged decisions unchanged.")
    ] = False,
) -> None:
    """Judge ambiguous aborted takes."""
    _configure_logging()
    words_path = project_dir / "artifacts" / "words.json"
    if not words_path.is_file():
        typer.echo(f"missing words artifact: {words_path}", err=True)
        raise typer.Exit(1)
    try:
        loaded = _resolve_profile(project_dir)
        artifact = run_judge(project_dir, loaded, force=force, no_llm=no_llm)
    except (FileNotFoundError, ValidationError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _echo_decisions(artifact.data.decisions, project_dir / "artifacts" / "decisions.json")


@app.command()
def fillers(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    profile: ProfileOpt = None,
    force: ForceOpt = False,
) -> None:
    """Mark filler words to drop.

    Writes ``artifacts/fillers.json``. Tighten does not drop these words yet.
    """
    artifacts = project_dir / "artifacts"
    for label, path in (
        ("words", artifacts / "words.json"),
        ("decisions", artifacts / "decisions.json"),
    ):
        if not path.is_file():
            typer.echo(f"missing {label} artifact: {path}", err=True)
            raise typer.Exit(1)
    try:
        loaded = _resolve_profile(project_dir, profile)
        artifact = run_fillers(project_dir, loaded, force=force)
    except (FileNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"filler decisions: {len(artifact.data.decisions)}")


@app.command()
def tighten(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
) -> None:
    """Place cut points and build the rough cut."""
    try:
        loaded = _resolve_profile(project_dir)
        artifact = run_tighten(project_dir, loaded, force=force)
    except (FileNotFoundError, ValidationError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    timeline_path = project_dir / "artifacts" / "timeline.json"
    typer.echo(
        f"{len(artifact.data.ranges)} ranges, {len(artifact.data.dropped)} dropped "
        f"-> {timeline_path}"
    )


@app.command()
def export(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
) -> None:
    """Write the FCPXML rough cut and rejects."""
    artifacts = project_dir / "artifacts"
    sources_path = artifacts / "sources.json"
    timeline_path = artifacts / "timeline.json"
    for path, label in ((sources_path, "sources"), (timeline_path, "timeline")):
        if not path.is_file():
            typer.echo(f"missing {label} artifact: {path}", err=True)
            raise typer.Exit(1)
    try:
        sources = SourcesArtifact.model_validate_json(sources_path.read_text(encoding="utf-8"))
        timeline = TimelineArtifact.model_validate_json(timeline_path.read_text(encoding="utf-8"))
        word_starts = _word_starts(artifacts / "words.json")
        result = export_fcpxml(
            sources.data,
            timeline.data,
            _resolve_profile(project_dir),
            project_dir.name,
            project_dir / "out" / f"{project_dir.name}.fcpxml",
            word_starts,
        )
    except (ValidationError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except FcpxmlValidationError as exc:
        typer.echo(f"DTD validation failed -> {exc.invalid_path}", err=True)
        raise typer.Exit(3) from exc
    checked = "DTD validated" if result.dtd_validated else "DTD not checked"
    typer.echo(f"wrote {result.path} ({checked})")


@app.command()
def run(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    profile: ProfileOpt = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip the judge.")] = False,
    force: ForceOpt = False,
    set_values: SetOpt = None,
) -> None:
    """Run every stage and write the rough cut."""
    _configure_logging()
    try:
        loaded = _resolve_profile(project_dir, profile, set_values)
        summary = run_project(project_dir, loaded, no_llm=no_llm, force=force)
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ToolMissing as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except FcpxmlValidationError as exc:
        typer.echo(f"DTD validation failed -> {exc.invalid_path}", err=True)
        raise typer.Exit(3) from exc
    except (FileNotFoundError, TimestampError, ValidationError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _echo_summary(summary)


@app.command("eval")
def evaluate(
    gold_dir: Annotated[Path, typer.Argument(help="Gold project folder.")],
    set_values: SetOpt = None,
) -> None:
    """Score a rough cut against a gold edit."""
    _configure_logging()
    try:
        load_profile("long", set_values)
        manual = _manual_fcpxml(gold_dir)
        project_dir = _eval_project(gold_dir)
        loaded = _resolve_profile(project_dir, overrides=set_values)
    except (ConfigError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    try:
        run_project(project_dir, loaded, no_llm=False, force=False)
        artifacts = project_dir / "artifacts"
        sources = SourcesArtifact.model_validate_json(
            (artifacts / "sources.json").read_text(encoding="utf-8")
        )
        words = WordsArtifact.model_validate_json(
            (artifacts / "words.json").read_text(encoding="utf-8")
        )
        timeline = TimelineArtifact.model_validate_json(
            (artifacts / "timeline.json").read_text(encoding="utf-8")
        )
        _metrics, table = evaluate_gold(
            words=words.data.words,
            sources=sources.data,
            timeline=timeline.data,
            manual_fcpxml=manual,
            eval_json=gold_dir / "eval.json",
        )
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ToolMissing as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except FcpxmlValidationError as exc:
        typer.echo(f"DTD validation failed -> {exc.invalid_path}", err=True)
        raise typer.Exit(3) from exc
    except (FileNotFoundError, TimestampError, ValidationError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(table)
    typer.echo(f"wrote {gold_dir / 'eval.json'}")


@app.command()
def words(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    from_s: Annotated[float, typer.Option("--from", help="Start time in seconds.")],
    to_s: Annotated[float, typer.Option("--to", help="End time in seconds.")],
) -> None:
    """Print words between two times."""
    path = project_dir / "artifacts" / "words.json"
    if not path.is_file():
        typer.echo(f"missing words artifact: {path}", err=True)
        raise typer.Exit(1)
    try:
        artifact = WordsArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(words_between(artifact.data.words, from_s, to_s))


def _word_starts(path: Path) -> dict[int, float] | None:
    """Word start times for CHECK markers. Absent words leave a marker at the cut."""
    if not path.is_file():
        return None
    artifact = WordsArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    return {word.i: word.start for word in artifact.data.words}


def _echo_audio(artifact: AudioEventsArtifact) -> None:
    typer.echo(f"speech segments: {len(artifact.data.speech)}, claps: {len(artifact.data.claps)}")


def _echo_alignment(artifact: AlignmentArtifact) -> None:
    data = artifact.data
    if data.script is None:
        typer.echo("no script")
        return
    typer.echo(
        f"sentences: {len(data.sentences)}, chapters: {len(data.chapters)}, "
        f"missing: {len(data.missing)}"
    )


def _echo_decisions(decisions: list[Decision], path: Path) -> None:
    dropped = sum(decision.action == "drop" for decision in decisions)
    flagged = sum(decision.flag for decision in decisions)
    typer.echo(f"{len(decisions)} decisions, {dropped} dropped, {flagged} flagged -> {path}")


def _echo_summary(summary: RunSummary) -> None:
    typer.echo(
        f"{summary.sources} sources, {summary.raw_duration_s:.3f}s raw, "
        f"{summary.output_duration_s:.3f}s output"
    )
    typer.echo(f"{summary.dropped} dropped, {summary.kept} kept, {summary.flagged} flagged")
    typer.echo(f"wrote {summary.fcpxml}")


def _resolve_profile(
    project_dir: Path,
    name: str | None = None,
    overrides: list[str] | None = None,
) -> Profile:
    """Load ``name``, or long/short from the frame when ``name`` is omitted.

    A project with nothing to measure yet keeps ``long``. The stage that
    follows reports the missing raw folder or artifact.
    """
    if name is not None or overrides:
        try:
            load_profile(name or "long", overrides)
        except ConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
    if name is not None:
        return load_profile(name, overrides)
    try:
        choice = choose_profile(project_dir)
    except FrameUnknown:
        return load_profile("long", overrides)
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ToolMissing as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"profile {choice.name} ({choice.orientation}, {choice.width}x{choice.height})")
    return load_profile(choice.name, overrides)


def _manual_fcpxml(gold_dir: Path) -> Path:
    candidates = [
        gold_dir / "manual.fcpxmld" / "Info.fcpxml",
        gold_dir / "manual.fcpxml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    joined = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing gold edit: {joined}")


def _eval_project(gold_dir: Path) -> Path:
    """Project whose raw sources the gold edit was cut from."""
    if (gold_dir / "raw").is_dir():
        return gold_dir
    sample = REPO_ROOT / "test" / "fixtures" / "sample"
    if (sample / "raw").is_dir():
        return sample
    raise FileNotFoundError(
        f"no raw sources for eval. Expected {gold_dir / 'raw'} or {sample / 'raw'}"
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

"""Cutter commands. Stages land one at a time; unimplemented ones say so."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from cutter.config import ConfigError, load_profile
from cutter.fcpxml import FcpxmlValidationError, export_fcpxml
from cutter.ingest import IngestError, ToolMissing, ingest_project
from cutter.models import SourcesArtifact, TimelineArtifact, WordsArtifact
from cutter.retakes import run_retakes
from cutter.transcribe import TimestampError, transcribe_project, words_between

app = typer.Typer(
    help="Make an editable rough cut from one project's English sources.",
    no_args_is_help=True,
)

ProfileOpt = Annotated[str, typer.Option("--profile", help="Profile name under profiles/.")]
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
    profile: ProfileOpt = "long",
    force: ForceOpt = False,
) -> None:
    """Validate sources and extract audio."""
    try:
        loaded = load_profile(profile)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    try:
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
        artifact = transcribe_project(project_dir, force=force)
    except (FileNotFoundError, TimestampError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    words_path = project_dir / "artifacts" / "words.json"
    typer.echo(f"transcribed {len(artifact.data.words)} words -> {words_path}")


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
        artifact = run_retakes(project_dir, force=force)
    except (FileNotFoundError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    decisions = artifact.data.decisions
    dropped = sum(decision.action == "drop" for decision in decisions)
    flagged = sum(decision.flag for decision in decisions)
    decisions_path = project_dir / "artifacts" / "decisions.json"
    typer.echo(
        f"{len(decisions)} decisions, {dropped} dropped, {flagged} flagged -> {decisions_path}"
    )


@app.command()
def judge(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Leave flagged decisions unchanged.")
    ] = False,
) -> None:
    """Judge ambiguous aborted takes."""
    _unimplemented("judge")


@app.command()
def tighten(
    project_dir: Annotated[Path, typer.Argument(help="Project folder.")],
    force: ForceOpt = False,
) -> None:
    """Place cut points and build the rough cut."""
    _unimplemented("tighten")


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
            load_profile(),
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
    profile: ProfileOpt = "long",
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip the judge.")] = False,
    force: ForceOpt = False,
    set_values: SetOpt = None,
) -> None:
    """Run every stage and write the rough cut."""
    _load(profile, set_values)
    _unimplemented("run")


@app.command("eval")
def evaluate(
    gold_dir: Annotated[Path, typer.Argument(help="Gold project folder.")],
    set_values: SetOpt = None,
) -> None:
    """Score a rough cut against a gold edit."""
    _load("long", set_values)
    _unimplemented("eval")


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


def _load(profile: str, overrides: list[str] | None) -> None:
    try:
        load_profile(profile, overrides)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _unimplemented(command: str) -> None:
    typer.echo(f"cutter {command} is not implemented yet.", err=True)
    raise typer.Exit(1)

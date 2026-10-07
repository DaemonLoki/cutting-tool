"""Pipeline orchestration. A judged decisions file stands in for retakes."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

from cutter.config import load_profile
from cutter.fcpxml import ExportResult
from cutter.models import (
    Source,
    SourcesArtifact,
    SourcesData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.retakes import run_retakes
from cutter.run import run_project
from cutter.tighten import run_tighten

FPS = "30000/1001"
PROFILE = load_profile("long")


def test_second_run_keeps_the_judged_decisions(tmp_path: Path, monkeypatch) -> None:
    calls = {"retakes": 0, "judge": 0, "export": 0}
    _patch(tmp_path, monkeypatch, calls)

    run_project(tmp_path, PROFILE, no_llm=True)
    decisions = (tmp_path / "artifacts" / "decisions.json").read_bytes()
    timeline = (tmp_path / "artifacts" / "timeline.json").read_bytes()
    assert b'"stage": "judge"' in decisions

    run_project(tmp_path, PROFILE, no_llm=True)

    assert calls == {"retakes": 1, "judge": 1, "export": 1}
    assert (tmp_path / "artifacts" / "decisions.json").read_bytes() == decisions
    assert (tmp_path / "artifacts" / "timeline.json").read_bytes() == timeline


def test_force_reruns_retakes_and_judge(tmp_path: Path, monkeypatch) -> None:
    calls = {"retakes": 0, "judge": 0, "export": 0}
    _patch(tmp_path, monkeypatch, calls)

    run_project(tmp_path, PROFILE, no_llm=True)
    run_project(tmp_path, PROFILE, no_llm=True, force=True)

    assert calls["retakes"] == 2
    assert calls["judge"] == 2
    assert calls["export"] == 2


def _patch(project_dir: Path, monkeypatch, calls: dict[str, int]) -> None:
    def ingest(directory: Path, profile, *, force: bool = False) -> SourcesData:
        path = directory / "artifacts" / "sources.json"
        if not path.is_file():
            _write_sources(directory)
        artifact = SourcesArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        return artifact.data

    def transcribe(directory: Path, *, profile=None, force: bool = False, transcriber=None):
        path = directory / "artifacts" / "words.json"
        if not path.is_file():
            _write_words(directory)
        return WordsArtifact.model_validate_json(path.read_text(encoding="utf-8"))

    def retakes(*args, **kwargs):
        calls["retakes"] += 1
        return run_retakes(*args, **kwargs)

    def judge(*args, **kwargs):
        calls["judge"] += 1
        from cutter.judge import run_judge

        return run_judge(*args, **kwargs)

    def export(sources, timeline, profile, project, out_path, word_starts=None):
        calls["export"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<fcpxml/>", encoding="utf-8")
        return ExportResult(path=out_path, dtd_validated=False, warning=None)

    monkeypatch.setattr("cutter.run.ingest_project", ingest)
    monkeypatch.setattr("cutter.run.transcribe_project", transcribe)
    monkeypatch.setattr("cutter.run.run_retakes", retakes)
    monkeypatch.setattr("cutter.run.run_judge", judge)
    monkeypatch.setattr("cutter.run.run_tighten", run_tighten)
    monkeypatch.setattr("cutter.run.export_fcpxml", export)


def _write_sources(project_dir: Path) -> None:
    wav_path = project_dir / "artifacts" / "audio" / "s01.48k.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, np.zeros(48_000, dtype=np.float32), 48_000)
    duration_s = 2.0
    duration_frames = math.floor(Fraction(duration_s) * Fraction(FPS))
    data = SourcesData(
        fps=FPS,
        width=1920,
        height=1080,
        audio_rate=48000,
        audio_channels=1,
        sources=[
            Source(
                id="s01",
                path=str(project_dir / "raw" / "s01.mov"),
                duration_s=duration_s,
                duration_frames=duration_frames,
                start_timecode="00:00:00:00",
                start_frames=0,
                vfr_warning=False,
                asr_wav="artifacts/audio/s01.16k.wav",
                analysis_wav="artifacts/audio/s01.48k.wav",
            )
        ],
    )
    write_artifact(
        project_dir / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=1,
                inputs_hash="sha256:sources",
                config_hash="sha256:sources-config",
            ),
            data=data,
        ),
    )


def _write_words(project_dir: Path) -> None:
    words = [
        Word(i=0, source="s01", w="Hello", norm="hello", start=0.1, end=0.4, sent=0),
        Word(i=1, source="s01", w="there.", norm="there", start=0.5, end=0.9, sent=0),
    ]
    write_artifact(
        project_dir / "artifacts" / "words.json",
        WordsArtifact(
            meta=make_meta(
                stage="transcribe",
                stage_version=1,
                inputs_hash="sha256:words",
                config_hash="sha256:words-config",
            ),
            data=WordsData(words=words),
        ),
    )

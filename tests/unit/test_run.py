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
    AlignmentArtifact,
    AlignmentData,
    AudioEventsArtifact,
    AudioEventsData,
    Clap,
    Decision,
    DecisionsArtifact,
    DecisionsData,
    ScriptChapter,
    ScriptInfo,
    ScriptSentence,
    Source,
    SourcesArtifact,
    SourcesData,
    SpeechSegment,
    Take,
    TimelineArtifact,
    TimelineChapter,
    TimelineData,
    Word,
    WordsArtifact,
    WordsData,
    WordSpan,
    make_meta,
    write_artifact,
)
from cutter.retakes import run_retakes
from cutter.run import _phase2_counts, run_project
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


def test_summary_reads_the_artifacts_just_written(tmp_path: Path, monkeypatch) -> None:
    calls = {"retakes": 0, "judge": 0, "export": 0}
    _patch(tmp_path, monkeypatch, calls)

    summary = run_project(tmp_path, PROFILE, no_llm=True)

    assert summary.speech_share == 0.0
    assert summary.claps == 0
    assert summary.filler_drops == 0
    assert summary.script_sentences_found == 0
    assert summary.script_sentences_missing == 0
    assert summary.unscripted_spans == 0
    assert summary.chapters == 0
    counts = _phase2_counts(tmp_path, summary.raw_duration_s)
    assert summary.speech_share == counts.speech_share
    assert summary.claps == counts.claps
    assert summary.chapters == counts.chapters


def test_phase2_counts_treat_a_missing_artifact_as_zero(tmp_path: Path) -> None:
    assert _phase2_counts(tmp_path, 10.0) == _phase2_counts(tmp_path, 0.0)
    empty = _phase2_counts(tmp_path, 10.0)
    assert empty.speech_share == 0.0
    assert empty.claps == 0
    assert empty.filler_drops == 0
    assert empty.script_sentences_found == 0
    assert empty.script_sentences_missing == 0
    assert empty.unscripted_spans == 0
    assert empty.chapters == 0


def test_phase2_counts_read_speech_script_fillers_and_chapters(tmp_path: Path) -> None:
    _write_phase2_artifacts(tmp_path)

    counts = _phase2_counts(tmp_path, 10.0)

    assert counts.speech_share == 0.25
    assert counts.claps == 1
    assert counts.filler_drops == 1
    assert counts.script_sentences_found == 1
    assert counts.script_sentences_missing == 1
    assert counts.unscripted_spans == 1
    assert counts.chapters == 2
    assert _phase2_counts(tmp_path, 0.0).speech_share == 0.0


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
    audio_dir = project_dir / "artifacts" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_dir / "s01.48k.wav", np.zeros(48_000, dtype=np.float32), 48_000)
    sf.write(audio_dir / "s01.16k.wav", np.zeros(16_000, dtype=np.float32), 16_000)
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


def _write_phase2_artifacts(project_dir: Path) -> None:
    artifacts = project_dir / "artifacts"
    meta = make_meta(
        stage="audio",
        stage_version=1,
        inputs_hash="sha256:phase2",
        config_hash="sha256:phase2-config",
    )
    write_artifact(
        artifacts / "audio_events.json",
        AudioEventsArtifact(
            meta=meta,
            data=AudioEventsData(
                backend="energy",
                speech=[
                    SpeechSegment(source="s01", start=1.0, end=2.0),
                    SpeechSegment(source="s01", start=4.0, end=5.5),
                ],
                claps=[Clap(source="s01", t=3.0, peak_db=-4.0, rise_db=22.0)],
            ),
        ),
    )
    write_artifact(
        artifacts / "fillers.json",
        DecisionsArtifact(
            meta=meta,
            data=DecisionsData(
                decisions=[
                    Decision(
                        id="f001",
                        kind="filler",
                        dropped_words=(0, 0),
                        kept_from_word=1,
                        match_words=0,
                        dropped_duration_s=0.2,
                        action="drop",
                        flag=False,
                    ),
                    Decision(
                        id="f002",
                        kind="filler",
                        dropped_words=(1, 1),
                        kept_from_word=2,
                        match_words=0,
                        dropped_duration_s=2.0,
                        action="keep",
                        flag=True,
                        flag_reason="filler_long",
                    ),
                ]
            ),
        ),
    )
    write_artifact(
        artifacts / "alignment.json",
        AlignmentArtifact(
            meta=meta,
            data=AlignmentData(
                script=ScriptInfo(path="script.md", hash="sha256:script"),
                chapters=[ScriptChapter(id="c01", title="Cache", level=1, first_sentence=0)],
                sentences=[
                    ScriptSentence(
                        id=0,
                        chapter="c01",
                        text="The cache stores the result.",
                        takes=[Take(first_word=0, last_word=4, score=100.0, chosen=True)],
                    ),
                    ScriptSentence(
                        id=1,
                        chapter=None,
                        text="This sentence is never spoken.",
                        takes=[],
                    ),
                ],
                unscripted=[WordSpan(first_word=5, last_word=6)],
                missing=[1],
            ),
        ),
    )
    write_artifact(
        artifacts / "timeline.json",
        TimelineArtifact(
            meta=meta,
            data=TimelineData(
                ranges=[],
                dropped=[],
                chapters=[
                    TimelineChapter(id="c01", title="Cache", at_word=0),
                    TimelineChapter(id="c02", title="Agent", at_word=5),
                ],
            ),
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

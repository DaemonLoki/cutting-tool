"""JSON artifacts written by each stage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactMeta(StrictModel):
    stage: str
    stage_version: int
    inputs_hash: str
    config_hash: str
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def utc_seconds(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).replace(microsecond=0)

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")


class Artifact[DataT: BaseModel](StrictModel):
    meta: ArtifactMeta
    data: DataT


def make_meta(
    *,
    stage: str,
    stage_version: int,
    inputs_hash: str,
    config_hash: str,
    created_at: datetime | None = None,
) -> ArtifactMeta:
    return ArtifactMeta(
        stage=stage,
        stage_version=stage_version,
        inputs_hash=inputs_hash,
        config_hash=config_hash,
        created_at=created_at or datetime.now(UTC),
    )


def write_artifact[DataT: BaseModel](path: Path, artifact: Artifact[DataT]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


class Source(StrictModel):
    id: str
    path: str
    duration_s: float
    duration_frames: int
    start_timecode: str
    start_frames: int
    vfr_warning: bool
    asr_wav: str
    analysis_wav: str
    proxy: str | None = None


class SourcesData(StrictModel):
    """Times elsewhere are seconds; frame counts here use the shared nominal rate."""

    fps: str
    width: int
    height: int
    audio_rate: int
    audio_channels: int
    sources: list[Source]


class SourcesArtifact(Artifact[SourcesData]):
    data: SourcesData


class Word(StrictModel):
    """One word. ``start`` and ``end`` are seconds from the start of its source."""

    i: int
    source: str
    w: str
    norm: str
    start: float
    end: float
    conf: float | None = None
    sent: int


class WordsData(StrictModel):
    words: list[Word]


class WordsArtifact(Artifact[WordsData]):
    data: WordsData


class SpeechSegment(StrictModel):
    """A run of voice in one source. Times are seconds from the start of that source."""

    source: str
    start: float
    end: float

    @model_validator(mode="after")
    def start_not_after_end(self) -> SpeechSegment:
        if self.start > self.end:
            raise ValueError("start must be <= end")
        return self


class Clap(StrictModel):
    """A broadband transient used as a mistake mark. ``t`` is the onset in seconds."""

    source: str
    t: float
    peak_db: float
    rise_db: float


class AudioEventsData(StrictModel):
    backend: Literal["silero", "energy", "none"]
    speech: list[SpeechSegment]
    claps: list[Clap]


class AudioEventsArtifact(Artifact[AudioEventsData]):
    data: AudioEventsData


class ScriptInfo(StrictModel):
    path: str
    hash: str


class Take(StrictModel):
    """One attempt at a script sentence. ``score`` is the match against that sentence."""

    first_word: int
    last_word: int
    score: float
    chosen: bool

    @model_validator(mode="after")
    def words_ordered(self) -> Take:
        if self.first_word > self.last_word:
            raise ValueError("first_word must be <= last_word")
        return self


class ScriptSentence(StrictModel):
    id: int
    chapter: str | None
    text: str
    takes: list[Take]


class ScriptChapter(StrictModel):
    id: str
    title: str
    level: int
    first_sentence: int


class WordSpan(StrictModel):
    """An inclusive run of transcript words."""

    first_word: int
    last_word: int

    @model_validator(mode="after")
    def words_ordered(self) -> WordSpan:
        if self.first_word > self.last_word:
            raise ValueError("first_word must be <= last_word")
        return self


class AlignmentData(StrictModel):
    script: ScriptInfo | None
    chapters: list[ScriptChapter]
    sentences: list[ScriptSentence]
    unscripted: list[WordSpan]
    missing: list[int]


class AlignmentArtifact(Artifact[AlignmentData]):
    data: AlignmentData


class JudgeVerdict(StrictModel):
    choice: Literal["A", "B", "both"]
    confidence: float
    reason: str
    model: str


class Decision(StrictModel):
    """Judgment on one candidate repeat: drop the failed take, or keep it for review."""

    id: str
    kind: Literal["retake", "filler"]
    dropped_words: tuple[int, int]
    kept_from_word: int
    match_words: int
    dropped_duration_s: float
    action: Literal["drop", "keep"]
    flag: bool
    flag_reason: str | None = None
    judge: JudgeVerdict | None = None
    evidence: Literal["transcript", "clap", "script"] = "transcript"
    clap_s: float | None = None
    score_gap: float | None = None

    @field_validator("dropped_words")
    @classmethod
    def inclusive_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        first, last = value
        if first > last:
            raise ValueError("dropped_words must be an inclusive range")
        return value


class DecisionsData(StrictModel):
    decisions: list[Decision]


class DecisionsArtifact(Artifact[DecisionsData]):
    data: DecisionsData


class Marker(StrictModel):
    at_word: int
    text: str


class Range(StrictModel):
    """A continuous span of one source that plays in the rough cut."""

    id: str
    source: str
    in_s: float
    out_s: float
    in_frame: int
    out_frame: int
    first_word: int
    last_word: int
    markers: list[Marker] = []


class DroppedSpan(StrictModel):
    source: str
    in_s: float
    out_s: float
    decision: str


class TimelineChapter(StrictModel):
    """A chapter marker. ``at_word`` is a kept word."""

    id: str
    title: str
    at_word: int


class TimelineData(StrictModel):
    ranges: list[Range]
    dropped: list[DroppedSpan]
    chapters: list[TimelineChapter] = []


class TimelineArtifact(Artifact[TimelineData]):
    data: TimelineData

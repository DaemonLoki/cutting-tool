"""JSON artifacts written by each stage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


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


class JudgeVerdict(StrictModel):
    choice: Literal["A", "B", "both"]
    confidence: float
    reason: str
    model: str


class Decision(StrictModel):
    """Judgment on one candidate repeat: drop the failed take, or keep it for review."""

    id: str
    kind: Literal["retake"]
    dropped_words: tuple[int, int]
    kept_from_word: int
    match_words: int
    dropped_duration_s: float
    action: Literal["drop", "keep"]
    flag: bool
    flag_reason: str | None = None
    judge: JudgeVerdict | None = None

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


class TimelineData(StrictModel):
    ranges: list[Range]
    dropped: list[DroppedSpan]


class TimelineArtifact(Artifact[TimelineData]):
    data: TimelineData

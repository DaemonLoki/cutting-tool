"""Round-trip every artifact through JSON."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    DroppedSpan,
    JudgeVerdict,
    Marker,
    Range,
    Source,
    SourcesArtifact,
    SourcesData,
    TimelineArtifact,
    TimelineData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
)


def _meta(stage: str):
    return make_meta(
        stage=stage,
        stage_version=1,
        inputs_hash="sha256:abc",
        config_hash="sha256:def",
        created_at=datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
    )


def _roundtrip(model):
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model
    return restored


def test_sources_roundtrip():
    artifact = SourcesArtifact(
        meta=_meta("ingest"),
        data=SourcesData(
            fps="30000/1001",
            width=3840,
            height=2160,
            audio_rate=48000,
            audio_channels=2,
            sources=[
                Source(
                    id="s01",
                    path="/abs/path/raw/01.mov",
                    duration_s=312.312,
                    duration_frames=9360,
                    start_timecode="00:00:00:00",
                    start_frames=0,
                    vfr_warning=False,
                    asr_wav="artifacts/audio/s01.16k.wav",
                    analysis_wav="artifacts/audio/s01.48k.wav",
                )
            ],
        ),
    )
    restored = _roundtrip(artifact)
    assert restored.model_dump(mode="json")["meta"]["created_at"] == "2026-10-06T12:00:00Z"


def test_words_roundtrip_keeps_null_confidence():
    artifact = WordsArtifact(
        meta=_meta("transcribe"),
        data=WordsData(
            words=[
                Word(
                    i=0,
                    source="s01",
                    w="So,",
                    norm="so",
                    start=1.20,
                    end=1.38,
                    conf=None,
                    sent=0,
                )
            ]
        ),
    )
    restored = _roundtrip(artifact)
    assert restored.data.words[0].conf is None
    assert restored.data.words[0].w == "So,"


def test_decisions_roundtrip_with_and_without_judge():
    judged = DecisionsArtifact(
        meta=_meta("judge"),
        data=DecisionsData(
            decisions=[
                Decision(
                    id="d001",
                    kind="retake",
                    dropped_words=(120, 141),
                    kept_from_word=142,
                    match_words=5,
                    dropped_duration_s=7.4,
                    action="drop",
                    flag=False,
                    flag_reason=None,
                    judge=JudgeVerdict(
                        choice="B",
                        confidence=0.82,
                        reason="The first attempt stops mid-sentence.",
                        model="qwen3-14b",
                    ),
                )
            ]
        ),
    )
    restored = _roundtrip(judged)
    payload = restored.model_dump(mode="json")
    assert payload["data"]["decisions"][0]["dropped_words"] == [120, 141]
    assert payload["data"]["decisions"][0]["judge"]["choice"] == "B"

    kept = Decision(
        id="d002",
        kind="retake",
        dropped_words=(10, 20),
        kept_from_word=21,
        match_words=4,
        dropped_duration_s=3.0,
        action="keep",
        flag=True,
        flag_reason="complete_sentence",
        judge=None,
    )
    assert _roundtrip(kept).judge is None


def test_timeline_roundtrip():
    artifact = TimelineArtifact(
        meta=_meta("tighten"),
        data=TimelineData(
            ranges=[
                Range(
                    id="r001",
                    source="s01",
                    in_s=1.12,
                    out_s=18.91,
                    in_frame=33,
                    out_frame=567,
                    first_word=0,
                    last_word=58,
                    markers=[Marker(at_word=12, text="CHECK: low-confidence retake (d003)")],
                )
            ],
            dropped=[DroppedSpan(source="s01", in_s=18.91, out_s=26.3, decision="d001")],
        ),
    )
    restored = _roundtrip(artifact)
    assert restored.data.ranges[0].markers[0].text.startswith("CHECK:")
    assert restored.data.dropped[0].decision == "d001"


def test_models_reject_unknown_keys():
    with pytest.raises(ValidationError):
        Word(
            i=0,
            source="s01",
            w="So",
            norm="so",
            start=0.0,
            end=0.2,
            sent=0,
            extra=1,
        )


def test_dropped_words_must_be_ordered():
    with pytest.raises(ValidationError):
        Decision(
            id="d001",
            kind="retake",
            dropped_words=(9, 2),
            kept_from_word=10,
            match_words=3,
            dropped_duration_s=1.0,
            action="drop",
            flag=False,
        )

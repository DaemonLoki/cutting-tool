"""Round-trip every artifact through JSON."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cutter.models import (
    AlignmentArtifact,
    AlignmentData,
    AudioEventsArtifact,
    AudioEventsData,
    Clap,
    Decision,
    DecisionsArtifact,
    DecisionsData,
    DroppedSpan,
    JudgeVerdict,
    Marker,
    Range,
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


def test_audio_events_roundtrip():
    artifact = AudioEventsArtifact(
        meta=_meta("audio"),
        data=AudioEventsData(
            backend="silero",
            speech=[SpeechSegment(source="s01", start=1.14, end=5.62)],
            claps=[Clap(source="s01", t=7.312, peak_db=-4.1, rise_db=31.0)],
        ),
    )
    restored = _roundtrip(artifact)
    assert restored.data.backend == "silero"
    assert restored.data.speech[0].end == 5.62
    assert restored.data.claps[0].t == 7.312


def test_alignment_roundtrip_with_and_without_script():
    take = Take(first_word=11, last_word=21, score=100.0, chosen=True)
    artifact = AlignmentArtifact(
        meta=_meta("align"),
        data=AlignmentData(
            script=ScriptInfo(path="script.md", hash="sha256:abc"),
            chapters=[ScriptChapter(id="c01", title="Intro", level=1, first_sentence=0)],
            sentences=[
                ScriptSentence(
                    id=0,
                    chapter="c01",
                    text="Do you want your designs to go from this to this?",
                    takes=[
                        Take(first_word=0, last_word=10, score=100.0, chosen=False),
                        take,
                    ],
                )
            ],
            unscripted=[WordSpan(first_word=40, last_word=55)],
            missing=[7],
        ),
    )
    restored = _roundtrip(artifact)
    assert restored.data.sentences[0].takes[1].chosen is True
    assert restored.data.missing == [7]

    empty = AlignmentData(
        script=None,
        chapters=[],
        sentences=[],
        unscripted=[],
        missing=[],
    )
    assert _roundtrip(empty).script is None


def test_filler_decision_and_timeline_chapter_roundtrip():
    decision = Decision(
        id="f001",
        kind="filler",
        dropped_words=(3, 3),
        kept_from_word=4,
        match_words=0,
        dropped_duration_s=0.4,
        action="drop",
        flag=False,
        evidence="transcript",
    )
    restored = _roundtrip(decision)
    assert restored.kind == "filler"
    assert restored.evidence == "transcript"
    assert restored.clap_s is None
    assert restored.score_gap is None

    chapter = TimelineChapter(id="c01", title="Intro", at_word=11)
    timeline = TimelineData(ranges=[], dropped=[], chapters=[chapter])
    assert _roundtrip(timeline).chapters[0].at_word == 11


def test_speech_and_word_spans_must_be_ordered():
    with pytest.raises(ValidationError):
        SpeechSegment(source="s01", start=2.0, end=1.0)
    with pytest.raises(ValidationError):
        Take(first_word=4, last_word=1, score=90.0, chosen=True)
    with pytest.raises(ValidationError):
        WordSpan(first_word=8, last_word=3)


def test_phase1_json_without_new_keys_still_validates():
    source = Source.model_validate(
        {
            "id": "s01",
            "path": "/tmp/01.mov",
            "duration_s": 1.0,
            "duration_frames": 30,
            "start_timecode": "00:00:00:00",
            "start_frames": 0,
            "vfr_warning": False,
            "asr_wav": "artifacts/audio/s01.16k.wav",
            "analysis_wav": "artifacts/audio/s01.48k.wav",
        }
    )
    assert source.proxy is None

    decision = Decision.model_validate(
        {
            "id": "d001",
            "kind": "retake",
            "dropped_words": [0, 2],
            "kept_from_word": 3,
            "match_words": 3,
            "dropped_duration_s": 1.2,
            "action": "drop",
            "flag": False,
        }
    )
    assert decision.evidence == "transcript"
    assert decision.clap_s is None
    assert decision.score_gap is None

    timeline = TimelineData.model_validate({"ranges": [], "dropped": []})
    assert timeline.chapters == []


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

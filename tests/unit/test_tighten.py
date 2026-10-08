"""Timeline ranges from synthetic words. Thresholds come from profile ``long``."""

import logging
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st

from cutter.config import load_profile
from cutter.models import (
    AlignmentData,
    AudioEventsArtifact,
    AudioEventsData,
    Clap,
    Decision,
    DecisionsArtifact,
    DecisionsData,
    DroppedSpan,
    Marker,
    ScriptChapter,
    ScriptSentence,
    Source,
    SourcesArtifact,
    SourcesData,
    SpeechSegment,
    Take,
    TimelineChapter,
    TimelineData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.tighten import STAGE_VERSION, build_timeline, run_tighten

FPS = "30000/1001"
# A dropped word of this length stays more than one frame away from the cuts beside it.
_MIN_WORD_MS = 200
PROFILE = load_profile("long")
TIGHTEN = PROFILE.tighten.model_copy(update={"min_range_frames": 1})


def _word(index: int, start: float, end: float, *, source: str = "s01") -> Word:
    token = f"w{index}"
    return Word(i=index, source=source, w=token, norm=token, start=start, end=end, sent=0)


def _sources(duration_s: float, *, source_id: str = "s01", fps: str = FPS) -> SourcesData:
    duration_frames = math.floor(Fraction(duration_s).limit_denominator(1_000_000) * Fraction(fps))
    return SourcesData(
        fps=fps,
        width=1920,
        height=1080,
        audio_rate=48000,
        audio_channels=1,
        sources=[
            Source(
                id=source_id,
                path=f"raw/{source_id}.mov",
                duration_s=duration_s,
                duration_frames=duration_frames,
                start_timecode="00:00:00:00",
                start_frames=0,
                vfr_warning=False,
                asr_wav=f"artifacts/audio/{source_id}.16k.wav",
                analysis_wav=f"artifacts/audio/{source_id}.48k.wav",
            )
        ],
    )


def _flat_envelope(duration_s: float) -> dict[str, tuple[np.ndarray, int]]:
    frame_ms = TIGHTEN.rms_frame_ms
    n_frames = math.ceil(duration_s * 1000 / frame_ms) + 2
    return {"s01": (np.ones(n_frames, dtype=np.float64), 48000)}


def _decision(
    number: int,
    first: int,
    last: int,
    words: list[Word],
    *,
    action: Literal["drop", "keep"] = "drop",
    flag: bool = False,
    flag_reason: str | None = None,
    kept_from_word: int | None = None,
) -> Decision:
    kept = last + 1 if last + 1 < len(words) else 0
    return Decision(
        id=f"d{number:03d}",
        kind="retake",
        dropped_words=(first, last),
        kept_from_word=kept if kept_from_word is None else kept_from_word,
        match_words=max(1, last - first + 1),
        dropped_duration_s=words[last].end - words[first].start,
        action=action,
        flag=flag,
        flag_reason=flag_reason,
    )


@st.composite
def _timelines(draw: st.DrawFn):
    count = draw(st.integers(min_value=1, max_value=8))
    gap = st.integers(min_value=0, max_value=50) | st.integers(min_value=1000, max_value=2500)
    cursor_ms = 0
    words: list[Word] = []
    for index in range(count):
        start_ms = cursor_ms + draw(gap)
        end_ms = start_ms + draw(st.integers(min_value=_MIN_WORD_MS, max_value=600))
        words.append(_word(index, start_ms / 1000, end_ms / 1000))
        cursor_ms = end_ms
    duration_s = (cursor_ms + draw(st.integers(min_value=500, max_value=2000))) / 1000
    dropped = draw(st.lists(st.booleans(), min_size=count, max_size=count))
    decisions: list[Decision] = []
    index = 0
    while index < count:
        if not dropped[index]:
            index += 1
            continue
        last = index
        while last + 1 < count and dropped[last + 1]:
            last += 1
        decisions.append(_decision(len(decisions) + 1, index, last, words))
        index = last + 1
    return words, decisions, _sources(duration_s), _flat_envelope(duration_s)


def _empty_alignment() -> AlignmentData:
    return AlignmentData(script=None, chapters=[], sentences=[], unscripted=[], missing=[])


def _assert_phase1_properties(words, decisions, timeline: TimelineData) -> None:
    dropped_ids: set[int] = set()
    for decision in decisions:
        if decision.action != "drop":
            continue
        first, last = decision.dropped_words
        dropped_ids.update(range(first, last + 1))
        for word in words[first : last + 1]:
            mid = (word.start + word.end) / 2
            assert all(not (rng.in_s < mid < rng.out_s) for rng in timeline.ranges)

    assert all(rng.in_frame < rng.out_frame for rng in timeline.ranges)
    ordered = sorted(timeline.ranges, key=lambda rng: (rng.source, rng.in_frame))
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        if prev.source == nxt.source:
            assert nxt.in_frame >= prev.out_frame

    for word in words:
        if word.i in dropped_ids:
            continue
        mid = (word.start + word.end) / 2
        assert any(rng.in_s <= mid <= rng.out_s for rng in timeline.ranges)


@settings(max_examples=50)
@given(_timelines())
def test_ranges_cover_kept_words_and_exclude_drops(case):
    words, decisions, sources, envelopes = case
    timeline = build_timeline(words, decisions, sources, envelopes, TIGHTEN)
    _assert_phase1_properties(words, decisions, timeline)
    empty = build_timeline(
        words,
        decisions,
        sources,
        envelopes,
        TIGHTEN,
        fillers=[],
        speech=[],
        claps=[],
        alignment=_empty_alignment(),
        vad=PROFILE.vad,
        claps_config=PROFILE.claps,
        script=PROFILE.script,
        chapters=PROFILE.chapters,
    )
    assert [rng.model_dump() for rng in empty.ranges] == [
        rng.model_dump() for rng in timeline.ranges
    ]
    _assert_phase1_properties(words, decisions, empty)


def test_gap_larger_than_max_gap_splits_ranges():
    config = load_profile("long").tighten
    gap_s = config.max_gap_ms / 1000
    words = [_word(0, 0.0, 0.2), _word(1, 0.2 + gap_s + 0.05, 0.9)]
    timeline = build_timeline(words, [], _sources(5.0), {}, config)

    assert [rng.id for rng in timeline.ranges] == ["r001", "r002"]
    assert timeline.ranges[0].last_word == 0
    assert timeline.ranges[1].first_word == 1


def test_dropped_decision_uses_word_boundaries_and_removes_those_words():
    words = [_word(0, 0.0, 0.4), _word(1, 0.5, 0.9), _word(2, 1.0, 1.4)]
    decision = _decision(1, 0, 1, words, kept_from_word=2)
    timeline = build_timeline(words, [decision], _sources(5.0), {}, load_profile("long").tighten)

    assert timeline.dropped == [
        DroppedSpan(source="s01", in_s=0.0, out_s=0.9, decision="d001")
    ]
    covered: set[int] = set()
    for rng in timeline.ranges:
        covered.update(range(rng.first_word, rng.last_word + 1))
    assert covered.isdisjoint({0, 1})
    assert 2 in covered


def test_flagged_decision_marks_the_range_containing_kept_from_word():
    words = [_word(0, 0.0, 0.4), _word(1, 0.5, 0.9), _word(2, 1.0, 1.4)]
    decision = _decision(
        1,
        0,
        0,
        words,
        action="keep",
        flag=True,
        flag_reason="long_segment",
        kept_from_word=1,
    )
    timeline = build_timeline(words, [decision], _sources(5.0), {}, load_profile("long").tighten)

    containing = [rng for rng in timeline.ranges if rng.first_word <= 1 <= rng.last_word]
    assert len(containing) == 1
    assert containing[0].markers == [Marker(at_word=1, text="CHECK: long_segment (d001)")]


def _speech_envelope(
    duration_s: float, voiced: list[tuple[float, float]]
) -> dict[str, tuple[np.ndarray, int]]:
    """10 ms RMS frames: -20 dB inside ``voiced`` spans, -60 dB elsewhere."""
    frame_s = TIGHTEN.rms_frame_ms / 1000
    count = math.ceil(duration_s / frame_s)
    envelope = np.full(count, 0.001, dtype=np.float64)
    for start, end in voiced:
        envelope[round(start / frame_s) : round(end / frame_s)] = 0.1
    return {"s01": (envelope, 48000)}


def test_range_ends_where_the_voice_of_a_stretched_last_word_stops():
    # "with." is stamped 25.04-26.00 in the sample, but the voice stops near
    # 25.66. The range must not keep that pause.
    config = load_profile("long").tighten
    words = [
        Word(i=0, source="s01", w="work", norm="work", start=24.72, end=25.04, sent=0),
        Word(i=1, source="s01", w="with.", norm="with", start=25.04, end=26.0, sent=0),
    ]
    envelopes = _speech_envelope(40.0, [(24.72, 25.66)])
    timeline = build_timeline(words, [], _sources(40.0), envelopes, config)

    assert len(timeline.ranges) == 1
    out_s = timeline.ranges[0].out_s
    assert out_s <= 25.66 + config.pad_tail_ms / 1000 + 0.03
    assert out_s < 26.0
    assert out_s >= (25.04 + 26.0) / 2


def test_out_point_never_snaps_past_the_tail_pad():
    config = load_profile("long").tighten
    words = [_word(0, 1.0, 1.5)]
    envelopes = _speech_envelope(10.0, [(1.0, 1.5)])
    timeline = build_timeline(words, [], _sources(10.0), envelopes, config)

    assert timeline.ranges[0].out_s <= 1.5 + config.pad_tail_ms / 1000 + 1e-9


def test_pause_hidden_in_a_sentence_end_starts_a_new_range():
    # The question's last word is stamped up to the next sentence, but its
    # voice stops 600 ms earlier, longer than max_gap_ms.
    config = load_profile("long").tighten
    words = [
        Word(i=0, source="s01", w="to", norm="to", start=11.12, end=11.44, sent=0),
        Word(i=1, source="s01", w="this?", norm="this", start=11.76, end=12.8, sent=0),
        Word(i=2, source="s01", w="Then", norm="then", start=13.12, end=13.36, sent=1),
    ]
    envelopes = _speech_envelope(20.0, [(11.12, 11.44), (11.76, 12.4), (13.12, 13.36)])
    timeline = build_timeline(words, [], _sources(20.0), envelopes, config)

    assert [(rng.first_word, rng.last_word) for rng in timeline.ranges] == [(0, 1), (2, 2)]
    assert timeline.ranges[0].out_s < 12.8


def test_tiny_fragment_far_from_neighbours_is_removed():
    config = load_profile("long").tighten.model_copy(update={"min_range_frames": 20})
    words = [_word(0, 0.0, 0.15), _word(1, 10.0, 14.0)]
    timeline = build_timeline(words, [], _sources(20.0), {}, config)

    assert len(timeline.ranges) == 1
    assert timeline.ranges[0].first_word == 1
    assert timeline.ranges[0].markers == [Marker(at_word=0, text="CHECK: tiny fragment removed")]


def test_run_tighten_cache_hit_keeps_the_file_bytes(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    wav_path = artifacts / "audio" / "s01.48k.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.zeros(48_000, dtype=np.float32), 48_000)
    words = [_word(0, 0.0, 0.4), _word(1, 0.5, 0.9)]
    meta = make_meta(
        stage="transcribe",
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
    )
    write_artifact(
        artifacts / "words.json",
        WordsArtifact(meta=meta, data=WordsData(words=words)),
    )
    write_artifact(
        artifacts / "decisions.json",
        DecisionsArtifact(meta=meta, data=DecisionsData(decisions=[])),
    )
    write_artifact(
        artifacts / "sources.json",
        SourcesArtifact(meta=meta.model_copy(update={"stage": "ingest"}), data=_sources(2.0)),
    )

    first = run_tighten(tmp_path)
    timeline_path = artifacts / "timeline.json"
    raw = timeline_path.read_bytes()
    time.sleep(1.1)
    second = run_tighten(tmp_path)

    assert second.data == first.data
    assert second.meta == first.meta
    assert timeline_path.read_bytes() == raw


def test_unscripted_threshold_misses_the_tighten_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifacts = tmp_path / "artifacts"
    wav_path = artifacts / "audio" / "s01.48k.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.zeros(48_000, dtype=np.float32), 48_000)
    words = [_word(0, 0.0, 0.4)]
    meta = make_meta(
        stage="transcribe",
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
    )
    write_artifact(artifacts / "words.json", WordsArtifact(meta=meta, data=WordsData(words=words)))
    write_artifact(
        artifacts / "decisions.json",
        DecisionsArtifact(meta=meta, data=DecisionsData(decisions=[])),
    )
    write_artifact(
        artifacts / "sources.json",
        SourcesArtifact(meta=meta.model_copy(update={"stage": "ingest"}), data=_sources(2.0)),
    )

    profile = load_profile("long")
    first = run_tighten(tmp_path, profile)
    changed = profile.model_copy(
        update={"script": profile.script.model_copy(update={"flag_unscripted_s": 1.0})}
    )
    second = run_tighten(tmp_path, changed)
    assert second.meta.inputs_hash == first.meta.inputs_hash
    assert second.meta.config_hash != first.meta.config_hash

    score_only = changed.model_copy(
        update={"script": changed.script.model_copy(update={"min_take_score": 40})}
    )

    def _unexpected_meta(**_kwargs: object) -> None:
        raise AssertionError("cache miss")

    monkeypatch.setattr("cutter.tighten.make_meta", _unexpected_meta)
    third = run_tighten(tmp_path, score_only)
    assert third.meta.created_at == second.meta.created_at


def _events(
    *,
    speech: list[SpeechSegment] | None = None,
    claps: list[Clap] | None = None,
    fillers: list[Decision] | None = None,
    alignment: AlignmentData | None = None,
    claps_config=None,
    chapters=None,
    script=None,
    vad=None,
):
    return {
        "fillers": fillers if fillers is not None else [],
        "speech": speech if speech is not None else [],
        "claps": claps if claps is not None else [],
        "alignment": alignment if alignment is not None else _empty_alignment(),
        "vad": PROFILE.vad if vad is None else vad,
        "claps_config": PROFILE.claps if claps_config is None else claps_config,
        "script": PROFILE.script if script is None else script,
        "chapters": PROFILE.chapters if chapters is None else chapters,
    }


def _overlaps_zone(rng, zone: tuple[float, float], fps: str) -> bool:
    if rng.in_s < zone[1] and rng.out_s > zone[0]:
        return True
    played_in = Fraction(rng.in_frame) / Fraction(fps)
    played_out = Fraction(rng.out_frame) / Fraction(fps)
    start = Fraction(zone[0]).limit_denominator(1_000_000)
    end = Fraction(zone[1]).limit_denominator(1_000_000)
    return played_in < end and played_out > start


def _filler(index: int, words: list[Word]) -> Decision:
    word = words[index]
    return Decision(
        id="f001",
        kind="filler",
        dropped_words=(index, index),
        kept_from_word=index + 1 if index + 1 < len(words) else 0,
        match_words=0,
        dropped_duration_s=word.end - word.start,
        action="drop",
        flag=False,
    )


def test_use_vad_false_and_empty_events_match_phase1_ranges():
    config = load_profile("long").tighten.model_copy(update={"use_vad": False})
    words = [
        Word(i=0, source="s01", w="to", norm="to", start=0.20, end=0.50, sent=0),
        Word(i=1, source="s01", w="this?", norm="this", start=0.60, end=1.40, sent=0),
        Word(i=2, source="s01", w="Then", norm="then", start=2.20, end=2.50, sent=1),
    ]
    decision = _decision(1, 0, 0, words, kept_from_word=1)
    sources = _sources(5.0)
    envelopes = _speech_envelope(5.0, [(0.20, 0.50), (0.60, 1.05), (2.20, 2.50)])
    phase1 = build_timeline(words, [decision], sources, envelopes, config)
    with_events = build_timeline(
        words,
        [decision],
        sources,
        envelopes,
        config,
        **_events(),
    )
    assert [rng.model_dump() for rng in with_events.ranges] == [
        rng.model_dump() for rng in phase1.ranges
    ]


def test_use_vad_false_ignores_speech_segments():
    config = TIGHTEN.model_copy(update={"use_vad": False})
    words = [_word(0, 1.0, 2.0)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    speech = [SpeechSegment(source="s01", start=0.8, end=1.2)]
    phase1 = build_timeline(words, [], sources, envelopes, config)
    ignored = build_timeline(words, [], sources, envelopes, config, speech=speech)
    assert [rng.model_dump() for rng in ignored.ranges] == [
        rng.model_dump() for rng in phase1.ranges
    ]


def test_vad_spoken_end_stops_at_the_segment():
    words = [Word(i=0, source="s01", w="with.", norm="with", start=1.0, end=2.0, sent=0)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    speech = [SpeechSegment(source="s01", start=0.95, end=1.6)]
    phase1 = build_timeline(words, [], sources, envelopes, TIGHTEN)
    vad = build_timeline(words, [], sources, envelopes, TIGHTEN, speech=speech)

    assert phase1.ranges[0].out_s >= 2.0
    assert vad.ranges[0].out_s <= 1.6 + TIGHTEN.pad_tail_ms / 1000 + 0.02
    assert vad.ranges[0].out_s < phase1.ranges[0].out_s


def test_vad_in_point_uses_a_segment_start_within_200ms():
    words = [_word(0, 1.0, 1.4)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    phase1 = build_timeline(words, [], sources, envelopes, TIGHTEN)
    pulled = build_timeline(
        words,
        [],
        sources,
        envelopes,
        TIGHTEN,
        speech=[SpeechSegment(source="s01", start=0.8, end=1.4)],
    )
    ignored = build_timeline(
        words,
        [],
        sources,
        envelopes,
        TIGHTEN,
        speech=[SpeechSegment(source="s01", start=0.7, end=1.4)],
    )

    assert pulled.ranges[0].in_s < phase1.ranges[0].in_s - 0.1
    assert ignored.ranges[0].in_s == phase1.ranges[0].in_s


def test_vad_gap_uses_spoken_end_for_every_kept_word():
    words = [_word(0, 1.0, 2.0), _word(1, 2.05, 2.30)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    speech = [SpeechSegment(source="s01", start=1.0, end=1.6)]
    phase1 = build_timeline(words, [], sources, envelopes, TIGHTEN)
    vad = build_timeline(words, [], sources, envelopes, TIGHTEN, speech=speech)

    assert [(rng.first_word, rng.last_word) for rng in phase1.ranges] == [(0, 1)]
    assert [(rng.first_word, rng.last_word) for rng in vad.ranges] == [(0, 0), (1, 1)]


def test_word_outside_every_segment_keeps_the_phase1_sentence_rule():
    words = [Word(i=0, source="s01", w="with.", norm="with", start=1.0, end=2.0, sent=0)]
    sources = _sources(5.0)
    envelopes = _speech_envelope(5.0, [(1.0, 1.4)])
    elsewhere = [SpeechSegment(source="s01", start=3.0, end=4.0)]
    phase1 = build_timeline(words, [], sources, envelopes, TIGHTEN)
    vad = build_timeline(words, [], sources, envelopes, TIGHTEN, speech=elsewhere)
    assert vad.ranges[0].out_s == phase1.ranges[0].out_s

    plain = [_word(0, 1.0, 2.0)]
    phase1_plain = build_timeline(plain, [], sources, envelopes, TIGHTEN)
    vad_plain = build_timeline(plain, [], sources, envelopes, TIGHTEN, speech=elsewhere)
    assert vad_plain.ranges[0].out_s > phase1_plain.ranges[0].out_s
    assert vad_plain.ranges[0].out_s >= 2.0


def test_in_point_never_starts_before_the_previous_spoken_end():
    # The dropped word's transcript runs to 2.0, but its voice stops at 1.9.
    # The next range may enter that tail and must not start before 1.9.
    words = [_word(0, 0.0, 0.30), _word(1, 0.40, 2.00), _word(2, 2.00, 2.40)]
    decision = _decision(1, 1, 1, words, kept_from_word=2)
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    speech = [
        SpeechSegment(source="s01", start=0.40, end=1.90),
        SpeechSegment(source="s01", start=1.90, end=2.40),
    ]
    phase1 = build_timeline(words, [decision], sources, envelopes, TIGHTEN)
    vad = build_timeline(words, [decision], sources, envelopes, TIGHTEN, speech=speech)

    assert vad.ranges[-1].first_word == 2
    assert vad.ranges[-1].in_s >= 1.90
    assert vad.ranges[-1].in_s < phase1.ranges[-1].in_s


def test_clap_zone_splits_a_range_and_stays_out_of_both_sides():
    words = [_word(0, 0.0, 0.20), _word(1, 0.45, 0.70)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    narrow = PROFILE.claps.model_copy(update={"exclude_before_ms": 20, "exclude_after_ms": 20})
    clap = Clap(source="s01", t=0.32, peak_db=-3.0, rise_db=30.0)
    zone = (0.30, 0.34)
    phase1 = build_timeline(words, [], sources, envelopes, TIGHTEN)
    timeline = build_timeline(
        words,
        [],
        sources,
        envelopes,
        TIGHTEN,
        claps=[clap],
        claps_config=narrow,
    )

    assert [(rng.first_word, rng.last_word) for rng in phase1.ranges] == [(0, 1)]
    assert [(rng.first_word, rng.last_word) for rng in timeline.ranges] == [(0, 0), (1, 1)]
    assert all(not _overlaps_zone(rng, zone, FPS) for rng in timeline.ranges)
    assert timeline.ranges[0].out_s <= zone[0]
    assert timeline.ranges[1].in_s >= zone[1]


def test_unavoidable_clap_is_kept_and_logged(caplog):
    words = [_word(0, 1.0, 1.6)]
    sources = _sources(5.0)
    envelopes = _flat_envelope(5.0)
    clap = Clap(source="s01", t=1.3, peak_db=-3.0, rise_db=30.0)
    with caplog.at_level(logging.WARNING, logger="cutter.tighten"):
        timeline = build_timeline(
            words,
            [],
            sources,
            envelopes,
            TIGHTEN,
            claps=[clap],
            claps_config=PROFILE.claps,
        )

    assert timeline.ranges[0].markers == [Marker(at_word=0, text="CHECK: clap inside range")]
    assert "CHECK: clap inside range" in caplog.text


def test_dropped_filler_midpoint_is_outside_every_range():
    words = [_word(0, 0.0, 0.30), _word(1, 0.40, 0.70), _word(2, 0.80, 1.10)]
    filler = _filler(1, words)
    timeline = build_timeline(
        words, [], _sources(5.0), _flat_envelope(5.0), TIGHTEN, fillers=[filler]
    )
    midpoint = (words[1].start + words[1].end) / 2
    assert all(not (rng.in_s < midpoint < rng.out_s) for rng in timeline.ranges)
    assert timeline.dropped == [
        DroppedSpan(source="s01", in_s=0.40, out_s=0.70, decision="f001")
    ]
    covered = {rng.first_word for rng in timeline.ranges}
    covered.update(rng.last_word for rng in timeline.ranges)
    assert covered >= {0, 2}


def test_chapter_at_word_is_the_first_kept_word_of_the_chosen_take():
    words = [_word(0, 0.0, 0.2), _word(1, 0.3, 0.5), _word(2, 0.6, 0.8), _word(3, 0.9, 1.1)]
    filler = _filler(2, words)
    alignment = AlignmentData(
        script=None,
        chapters=[ScriptChapter(id="c01", title="Intro", level=1, first_sentence=0)],
        sentences=[
            ScriptSentence(id=0, chapter="c01", text="Missing.", takes=[]),
            ScriptSentence(
                id=1,
                chapter="c01",
                text="Kept line.",
                takes=[Take(first_word=2, last_word=3, score=100.0, chosen=True)],
            ),
        ],
        unscripted=[],
        missing=[0],
    )
    timeline = build_timeline(
        words,
        [],
        _sources(5.0),
        _flat_envelope(5.0),
        TIGHTEN,
        fillers=[filler],
        alignment=alignment,
        chapters=PROFILE.chapters,
    )
    assert timeline.chapters == [TimelineChapter(id="c01", title="Intro", at_word=3)]
    dropped = set(range(filler.dropped_words[0], filler.dropped_words[1] + 1))
    assert timeline.chapters[0].at_word not in dropped


def test_chapter_without_a_chosen_take_is_skipped(caplog):
    words = [_word(0, 0.0, 0.4)]
    alignment = AlignmentData(
        script=None,
        chapters=[ScriptChapter(id="c01", title="Intro", level=1, first_sentence=0)],
        sentences=[ScriptSentence(id=0, chapter="c01", text="Missing.", takes=[])],
        unscripted=[],
        missing=[0],
    )
    with caplog.at_level(logging.WARNING, logger="cutter.tighten"):
        timeline = build_timeline(
            words,
            [],
            _sources(5.0),
            {},
            TIGHTEN,
            alignment=alignment,
            chapters=PROFILE.chapters,
        )
    assert timeline.chapters == []
    assert "skipping" in caplog.text


def test_unscripted_span_marks_its_first_kept_word():
    words = [_word(0, 0.0, 5.0), _word(1, 5.0, 25.0), _word(2, 30.0, 31.0)]
    long_span = AlignmentData(
        script=None,
        chapters=[],
        sentences=[],
        unscripted=[{"first_word": 0, "last_word": 1}],
        missing=[],
    )
    # WordSpan is a model; construct it properly if dict validation works via AlignmentData.
    short = AlignmentData(
        script=None,
        chapters=[],
        sentences=[],
        unscripted=[{"first_word": 2, "last_word": 2}],
        missing=[],
    )
    flagged = build_timeline(
        words,
        [],
        _sources(40.0),
        {},
        TIGHTEN,
        alignment=long_span,
        script=PROFILE.script,
    )
    quiet = build_timeline(
        words,
        [],
        _sources(40.0),
        {},
        TIGHTEN,
        alignment=short,
        script=PROFILE.script,
    )
    assert any(
        marker.text == "CHECK: unscripted" and marker.at_word == 0
        for rng in flagged.ranges
        for marker in rng.markers
    )
    assert all(
        marker.text != "CHECK: unscripted" for rng in quiet.ranges for marker in rng.markers
    )


def test_unheard_speech_marks_the_next_range_and_is_not_cut_in():
    words = [_word(0, 5.0, 5.4)]
    sources = _sources(20.0)
    long = SpeechSegment(source="s01", start=0.0, end=3.0)
    short = SpeechSegment(source="s01", start=0.0, end=0.4)
    at_end = SpeechSegment(source="s01", start=10.0, end=13.0)
    marked = build_timeline(
        words, [], sources, {}, TIGHTEN, speech=[long], vad=PROFILE.vad
    )
    ignored = build_timeline(
        words, [], sources, {}, TIGHTEN, speech=[short], vad=PROFILE.vad
    )
    trailing = build_timeline(
        words, [], sources, {}, TIGHTEN, speech=[at_end], vad=PROFILE.vad
    )

    assert marked.ranges[0].markers == [
        Marker(at_word=0, text="CHECK: speech without transcript")
    ]
    assert marked.ranges[0].in_s > 3.0 or marked.ranges[0].first_word == 0
    assert all(rng.out_s - rng.in_s < 3 for rng in marked.ranges)
    assert ignored.ranges[0].markers == []
    assert trailing.ranges[0].markers == [
        Marker(at_word=0, text="CHECK: speech without transcript")
    ]


@settings(max_examples=30)
@given(_timelines())
def test_clap_zones_and_filler_drops_stay_out_of_ranges(case):
    words, decisions, sources, envelopes = case
    dropped_ids = {index for decision in decisions for index in range(*_span(decision))}
    kept = [word for word in words if word.i not in dropped_ids]
    fillers = [_filler(kept[0].i, words)] if kept else []
    last = words[-1]
    zone_start = last.end + TIGHTEN.min_tail_ms / 1000 + 0.02
    clap_t = zone_start + PROFILE.claps.exclude_before_ms / 1000
    zone = (zone_start, clap_t + PROFILE.claps.exclude_after_ms / 1000)
    clap = Clap(source="s01", t=clap_t, peak_db=-3.0, rise_db=30.0)
    timeline = build_timeline(
        words,
        decisions,
        sources,
        envelopes,
        TIGHTEN,
        fillers=fillers,
        claps=[clap],
        claps_config=PROFILE.claps,
    )
    assert all(not _overlaps_zone(rng, zone, sources.fps) for rng in timeline.ranges)
    if fillers:
        filler = fillers[0]
        word = words[filler.dropped_words[0]]
        midpoint = (word.start + word.end) / 2
        assert all(not (rng.in_s < midpoint < rng.out_s) for rng in timeline.ranges)
        assert filler.dropped_words[0] not in {
            index
            for rng in timeline.ranges
            for index in range(rng.first_word, rng.last_word + 1)
        }


def _span(decision: Decision) -> tuple[int, int]:
    first, last = decision.dropped_words
    return first, last + 1


def test_run_tighten_reloads_when_the_stage_version_or_fillers_change(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    wav_path = artifacts / "audio" / "s01.48k.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.zeros(48_000, dtype=np.float32), 48_000)
    words = [_word(0, 0.0, 0.4), _word(1, 0.5, 0.9)]
    meta = make_meta(
        stage="transcribe",
        stage_version=1,
        inputs_hash="sha256:in",
        config_hash="sha256:cfg",
    )
    write_artifact(artifacts / "words.json", WordsArtifact(meta=meta, data=WordsData(words=words)))
    write_artifact(
        artifacts / "decisions.json",
        DecisionsArtifact(meta=meta, data=DecisionsData(decisions=[])),
    )
    write_artifact(
        artifacts / "sources.json",
        SourcesArtifact(meta=meta.model_copy(update={"stage": "ingest"}), data=_sources(2.0)),
    )

    first = run_tighten(tmp_path)
    assert first.meta.stage_version == STAGE_VERSION
    timeline_path = artifacts / "timeline.json"
    stale = first.model_copy(update={"meta": first.meta.model_copy(update={"stage_version": 3})})
    write_artifact(timeline_path, stale)
    rebuilt = run_tighten(tmp_path)
    assert rebuilt.meta.stage_version == STAGE_VERSION

    write_artifact(
        artifacts / "fillers.json",
        DecisionsArtifact(
            meta=meta.model_copy(update={"stage": "fillers"}),
            data=DecisionsData(decisions=[_filler(0, words)]),
        ),
    )
    write_artifact(
        artifacts / "audio_events.json",
        AudioEventsArtifact(
            meta=meta.model_copy(update={"stage": "audio"}),
            data=AudioEventsData(backend="none", speech=[], claps=[]),
        ),
    )
    with_fillers = run_tighten(tmp_path)
    assert with_fillers.meta.inputs_hash != rebuilt.meta.inputs_hash
    midpoint = (words[0].start + words[0].end) / 2
    assert all(not (rng.in_s < midpoint < rng.out_s) for rng in with_fillers.data.ranges)
    assert with_fillers.data.dropped[0].decision == "f001"

"""Timeline ranges from synthetic words. Thresholds come from profile ``long``."""

import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st

from cutter.config import load_profile
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    DroppedSpan,
    Marker,
    Source,
    SourcesArtifact,
    SourcesData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.tighten import build_timeline, run_tighten

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


@settings(max_examples=50)
@given(_timelines())
def test_ranges_cover_kept_words_and_exclude_drops(case):
    words, decisions, sources, envelopes = case
    timeline = build_timeline(words, decisions, sources, envelopes, TIGHTEN)
    dropped_ids: set[int] = set()
    for decision in decisions:
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

"""Gold-edit scoring: false cuts, missed retakes, and eval.json."""

from __future__ import annotations

from pathlib import Path

from cutter.evaluate import diff_metrics, evaluate_gold, load_metrics, predicted_spans, score
from cutter.fcpxml_parse import SourceSpan
from cutter.models import Range, Source, SourcesData, TimelineData, Word


def _word(
    i: int,
    start: float,
    end: float,
    sent: int,
    source: str = "s1",
) -> Word:
    return Word(i=i, source=source, w=f"w{i}", norm=f"w{i}", start=start, end=end, sent=sent)


def _sources() -> SourcesData:
    return SourcesData(
        fps="24000/1001",
        width=3840,
        height=2160,
        audio_rate=48000,
        audio_channels=2,
        sources=[
            Source(
                id="s1",
                path="/videos/1-intro.MP4",
                duration_s=1000.0,
                duration_frames=24000,
                start_timecode="00:00:00:00",
                start_frames=0,
                vfr_warning=False,
                asr_wav="s1.wav",
                analysis_wav="s1-analysis.wav",
            )
        ],
    )


def _timeline(*ranges: tuple[float, float]) -> TimelineData:
    return TimelineData(
        ranges=[
            Range(
                id=f"r{index}",
                source="s1",
                in_s=in_s,
                out_s=out_s,
                in_frame=0,
                out_frame=1,
                first_word=0,
                last_word=0,
            )
            for index, (in_s, out_s) in enumerate(ranges)
        ],
        dropped=[],
    )


def _gold_fcpxml(tmp_path: Path, *, start: str = "0s", duration: str = "10s") -> Path:
    path = tmp_path / "Info.fcpxml"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.14">
  <resources>
    <asset id="r2" name="1-intro" start="0s" duration="1000s" hasVideo="1" hasAudio="1">
      <media-rep kind="original-media" src="file:///videos/1-intro.MP4"/>
    </asset>
  </resources>
  <library>
    <event name="day">
      <project name="demo">
        <sequence duration="{duration}" tcStart="0s">
          <spine>
            <asset-clip ref="r2" offset="0s" name="1-intro" start="{start}" duration="{duration}"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
""",
        encoding="utf-8",
    )
    return path


def test_matching_spans_score_perfectly():
    words = [_word(0, 0.0, 1.0, 0), _word(1, 1.0, 2.0, 0)]
    spans = [SourceSpan("1-intro.MP4", 0.0, 5.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, spans, spans)
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.false_cut_sentences == 0
    assert metrics.missed_retake_words == 0
    assert metrics.kept_words == 2
    assert metrics.gold_kept_words == 2


def test_dropped_gold_sentence_is_a_false_cut():
    words = [
        _word(0, 0.0, 1.0, 0),
        _word(1, 1.0, 2.0, 0),
        _word(2, 3.0, 4.0, 1),
    ]
    gold = [SourceSpan("1-intro.MP4", 0.0, 10.0)]
    predicted = [SourceSpan("1-intro.MP4", 1.0, 10.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, gold, predicted)
    assert metrics.false_cut_sentences == 1
    assert metrics.recall == 2 / 3
    assert metrics.recall < 1.0


def test_kept_failed_take_is_a_missed_retake():
    words = [_word(0, 0.0, 1.0, 0), _word(1, 3.0, 4.0, 1)]
    gold = [SourceSpan("1-intro.MP4", 0.0, 2.0)]
    predicted = [SourceSpan("1-intro.MP4", 0.0, 2.0), SourceSpan("1-intro.MP4", 3.0, 5.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, gold, predicted)
    assert metrics.missed_retake_words == 1
    assert metrics.precision == 0.5
    assert metrics.precision < 1.0


def test_missed_retake_rate_per_10_minutes():
    words = [_word(0, 1000.0, 1001.0, 0), _word(1, 1001.0, 1002.0, 0)]
    gold = [SourceSpan("1-intro.MP4", 0.0, 600.0)]
    predicted = [SourceSpan("1-intro.MP4", 1000.0, 1010.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, gold, predicted)
    assert metrics.missed_retake_words == 2
    assert metrics.gold_duration_s == 600.0
    assert metrics.missed_retake_words_per_10min == 2.0


def test_midpoint_on_out_s_is_outside_the_span():
    words = [
        _word(0, 8.0, 10.0, 0),
        _word(1, 9.0, 11.0, 1),
    ]
    spans = [SourceSpan("1-intro.MP4", 0.0, 10.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, spans, spans)
    assert metrics.gold_kept_words == 1
    assert metrics.kept_words == 1
    assert metrics.false_cut_sentences == 0
    assert metrics.missed_retake_words == 0


def test_unknown_source_is_dropped_on_both_sides():
    words = [_word(0, 0.0, 1.0, 0), _word(1, 0.0, 1.0, 1, source="missing")]
    spans = [SourceSpan("1-intro.MP4", 0.0, 5.0)]
    metrics = score(words, {"s1": "1-intro.MP4"}, spans, spans)
    assert metrics.f1 == 1.0
    assert metrics.gold_kept_words == 1
    assert metrics.kept_words == 1
    assert metrics.false_cut_sentences == 0
    assert metrics.missed_retake_words == 0


def test_predicted_spans_use_the_source_filename():
    assert predicted_spans(_timeline((1.0, 4.0)), _sources()) == [
        SourceSpan("1-intro.MP4", 1.0, 4.0)
    ]


def test_evaluate_gold_writes_eval_and_second_call_shows_delta(tmp_path: Path):
    manual = _gold_fcpxml(tmp_path)
    eval_json = tmp_path / "nested" / "eval.json"
    words = [_word(0, 0.0, 1.0, 0), _word(1, 1.0, 2.0, 0)]
    sources = _sources()
    assert load_metrics(eval_json) is None

    metrics, table = evaluate_gold(
        words=words,
        sources=sources,
        timeline=_timeline((0.0, 10.0)),
        manual_fcpxml=manual,
        eval_json=eval_json,
    )
    assert metrics.f1 == 1.0
    assert metrics.false_cut_sentences == 0
    assert load_metrics(eval_json) == metrics
    assert "delta" not in table

    later, delta_table = evaluate_gold(
        words=words,
        sources=sources,
        timeline=_timeline(),
        manual_fcpxml=manual,
        eval_json=eval_json,
    )
    assert later.recall < metrics.recall
    assert later.false_cut_sentences == 1
    assert load_metrics(eval_json) == later
    assert "delta" in delta_table
    assert "-" in delta_table
    assert diff_metrics(metrics, later) == delta_table

"""Script decisions and clap anchors on top of the Phase 1 transcript scan."""

import logging
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from cutter.cache import inputs_hash
from cutter.config import load_profile
from cutter.models import (
    AlignmentArtifact,
    AlignmentData,
    AudioEventsArtifact,
    AudioEventsData,
    Clap,
    Decision,
    ScriptInfo,
    ScriptSentence,
    Take,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.retakes import detect_retakes, run_retakes

PROFILE = load_profile("long")
LONG = PROFILE.retakes
CLAPS = PROFILE.claps
SCRIPT = PROFILE.script
_TOKENS = ("the", "agent", "joins", "call", "server", "vote.", "guess.", "and", "now")


def _norm(token: str) -> str:
    return "".join(ch for ch in token.casefold() if ch.isalnum())


def _words(
    tokens: list[str],
    *,
    source: str = "01",
    start: float = 0.0,
    step: float = 0.4,
    duration: float = 0.32,
    starts: list[float] | None = None,
) -> list[Word]:
    words: list[Word] = []
    for offset, token in enumerate(tokens):
        begin = starts[offset] if starts is not None else start + offset * step
        words.append(
            Word(
                i=offset,
                source=source,
                w=token,
                norm=_norm(token),
                start=begin,
                end=begin + duration,
                sent=0,
            )
        )
    return words


def _indexed(*groups: list[Word]) -> list[Word]:
    words: list[Word] = []
    sent = 0
    for group in groups:
        for word in group:
            words.append(word.model_copy(update={"i": len(words), "sent": sent}))
            if word.w.endswith((".", "?", "!")):
                sent += 1
    return words


def _clap(source: str, t: float) -> Clap:
    return Clap(source=source, t=t, peak_db=-4.0, rise_db=30.0)


def _alignment(sentences: list[ScriptSentence]) -> AlignmentData:
    return AlignmentData(
        script=ScriptInfo(path="script.md", hash="sha256:abc"),
        chapters=[],
        sentences=sentences,
        unscripted=[],
        missing=[],
    )


def _neuron_pair() -> list[Word]:
    first = ["A", "neuron", "is", "a", "weighted", "vote."]
    second = ["A", "neuron", "is", "a", "weighted", "guess."]
    return _indexed(_words(first + second))


def _neuron_takes(chosen_score: float, alternate_score: float) -> list[Take]:
    return [
        Take(first_word=0, last_word=5, score=chosen_score, chosen=True),
        Take(first_word=6, last_word=11, score=alternate_score, chosen=False),
    ]


def _write_words(project: Path, words: list[Word]) -> None:
    write_artifact(
        project / "artifacts" / "words.json",
        WordsArtifact(
            meta=make_meta(
                stage="transcribe",
                stage_version=1,
                inputs_hash="sha256:in",
                config_hash="sha256:cfg",
            ),
            data=WordsData(words=words),
        ),
    )


def _assert_disjoint(decisions: list[Decision]) -> None:
    spans = [decision.dropped_words for decision in decisions]
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            assert left[0] > right[1] or right[0] > left[1]
    assert [decision.kind for decision in decisions] == ["retake"] * len(decisions)
    expected = [f"d{number:03d}" for number in range(1, len(decisions) + 1)]
    assert [decision.id for decision in decisions] == expected


def test_empty_alignment_and_no_claps_match_the_transcript_scan():
    words = _indexed(
        _words(["the", "agent", "joins", "the", "the", "agent", "joins", "the", "call"])
    )
    empty = AlignmentData(script=None, chapters=[], sentences=[], unscripted=[], missing=[])
    found = detect_retakes(
        words,
        LONG,
        claps=[],
        claps_config=CLAPS,
        alignment=empty,
        script_config=SCRIPT,
    )
    assert found == detect_retakes(words, LONG)
    _assert_disjoint(found)


def test_clap_drops_an_asr_smoothed_restart():
    assert LONG.min_match_words == 3
    assert CLAPS.min_match_words == 2
    words = _indexed(
        _words(["the", "agent", "the", "agent", "joins"], starts=[0.0, 0.4, 1.2, 1.6, 2.0])
    )
    assert detect_retakes(words, LONG) == []

    decisions = detect_retakes(words, LONG, claps=[_clap("01", 0.9)], claps_config=CLAPS)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 1)
    assert decision.kept_from_word == 2
    assert decision.match_words == 2
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.evidence == "clap"
    assert decision.clap_s == 0.9
    assert decision.score_gap is None
    assert decision.kind == "retake"


def test_disabled_claps_leave_the_transcript_scan_alone():
    words = _indexed(
        _words(["the", "agent", "the", "agent", "joins"], starts=[0.0, 0.4, 1.2, 1.6, 2.0])
    )
    disabled = CLAPS.model_copy(update={"enabled": False})
    assert detect_retakes(words, LONG, claps=[_clap("01", 0.9)], claps_config=disabled) == []


def test_clap_without_a_retake_keeps_the_span_from_the_sentence_end():
    words = _indexed(
        _words(
            ["Hello", "there.", "purple", "badger", "waddles", "Next", "chapter."],
            starts=[0.0, 0.4, 0.8, 1.2, 1.6, 2.4, 2.8],
        )
    )
    decisions = detect_retakes(words, LONG, claps=[_clap("01", 2.0)], claps_config=CLAPS)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (2, 4)
    assert decision.kept_from_word == 5
    assert decision.action == "keep"
    assert decision.flag is True
    assert decision.flag_reason == "clap_without_retake"
    assert decision.evidence == "clap"
    assert decision.clap_s == 2.0
    _assert_disjoint(decisions)


def test_second_clap_starts_after_the_previous_clap():
    words = _indexed(
        _words(
            ["Done.", "alpha", "beta", "gamma", "delta", "omega"],
            starts=[0.0, 0.5, 0.9, 1.4, 1.8, 2.6],
        )
    )
    decisions = detect_retakes(
        words,
        LONG,
        claps=[_clap("01", 1.15), _clap("01", 2.2)],
        claps_config=CLAPS,
    )

    assert [decision.dropped_words for decision in decisions] == [(1, 2), (3, 4)]
    assert all(decision.action == "keep" for decision in decisions)
    assert all(decision.flag_reason == "clap_without_retake" for decision in decisions)
    assert all(decision.evidence == "clap" for decision in decisions)
    _assert_disjoint(decisions)


def test_clap_span_stays_inside_its_source():
    words = _indexed(
        _words(["Done.", "alpha"], source="01", starts=[0.0, 0.4]),
        _words(["beta", "gamma", "later"], source="02", starts=[0.0, 0.4, 1.0]),
    )
    decisions = detect_retakes(words, LONG, claps=[_clap("02", 0.7)], claps_config=CLAPS)

    assert len(decisions) == 1
    assert decisions[0].dropped_words == (2, 3)
    assert decisions[0].kept_from_word == 4
    assert words[decisions[0].dropped_words[0]].source == "02"


def test_clap_after_a_finished_sentence_records_nothing(caplog) -> None:
    words = _indexed(_words(["Done.", "Next."], starts=[0.0, 0.8]))
    with caplog.at_level(logging.INFO, logger="cutter.retakes"):
        decisions = detect_retakes(words, LONG, claps=[_clap("01", 0.5)], claps_config=CLAPS)
    assert decisions == []
    assert "no words before it" in caplog.text


def test_script_drops_the_later_take_when_the_gap_is_at_least_5():
    words = _neuron_pair()
    plain = detect_retakes(words, LONG)
    assert plain[0].action == "keep"
    assert plain[0].flag_reason == "complete_sentence"
    assert plain[0].dropped_words == (0, 5)

    alignment = _alignment(
        [
            ScriptSentence(
                id=0,
                chapter=None,
                text="A neuron is a weighted vote.",
                takes=_neuron_takes(100.0, 95.0),
            )
        ]
    )
    decisions = detect_retakes(words, LONG, alignment=alignment, script_config=SCRIPT)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (6, 11)
    assert decision.kept_from_word == 0
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.flag_reason is None
    assert decision.evidence == "script"
    assert decision.score_gap == 5
    assert decision.clap_s is None
    assert decision.kind == "retake"
    _assert_disjoint(decisions)


def test_script_flags_a_close_call_when_the_gap_is_under_5():
    words = _neuron_pair()
    alignment = _alignment(
        [
            ScriptSentence(
                id=0,
                chapter=None,
                text="A neuron is a weighted vote.",
                takes=_neuron_takes(100.0, 96.0),
            )
        ]
    )
    decisions = detect_retakes(words, LONG, alignment=alignment, script_config=SCRIPT)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (6, 11)
    assert decision.action == "drop"
    assert decision.flag is True
    assert decision.flag_reason == "script_close_call"
    assert decision.evidence == "script"
    assert decision.score_gap == 4


def test_script_drops_an_earlier_finished_sentence_without_phase1_guards():
    words = _neuron_pair()
    takes = [
        Take(first_word=0, last_word=5, score=80.0, chosen=False),
        Take(first_word=6, last_word=11, score=100.0, chosen=True),
    ]
    decisions = detect_retakes(
        words,
        LONG,
        alignment=_alignment(
            [ScriptSentence(id=0, chapter=None, text="A neuron is a weighted guess.", takes=takes)]
        ),
        script_config=SCRIPT,
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 5)
    assert decision.kept_from_word == 6
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.evidence == "script"
    assert decision.score_gap == 20


def test_script_keeps_a_later_alternate_when_drop_later_takes_is_false():
    words = _neuron_pair()
    keeping = SCRIPT.model_copy(update={"drop_later_takes": False})
    alignment = _alignment(
        [
            ScriptSentence(
                id=0,
                chapter=None,
                text="A neuron is a weighted vote.",
                takes=_neuron_takes(100.0, 80.0),
            )
        ]
    )
    decisions = detect_retakes(words, LONG, alignment=alignment, script_config=keeping)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (6, 11)
    assert decision.kept_from_word == 0
    assert decision.action == "keep"
    assert decision.flag is True
    assert decision.flag_reason == "script_close_call"
    assert decision.evidence == "script"
    assert decision.score_gap == 20


def test_script_with_no_alternates_matches_the_transcript_scan():
    words = _neuron_pair()
    alignment = _alignment(
        [
            ScriptSentence(
                id=0,
                chapter=None,
                text="A neuron is a weighted vote.",
                takes=[Take(first_word=0, last_word=5, score=100.0, chosen=True)],
            )
        ]
    )
    assert detect_retakes(words, LONG, alignment=alignment, script_config=SCRIPT) == detect_retakes(
        words, LONG
    )


def test_short_tail_after_an_alternate_joins_the_dropped_span():
    words = _indexed(
        _words(
            ["the", "agent", "joins", "um", "no", "A", "neuron", "is", "a", "weighted", "vote."]
        )
    )
    takes = [
        Take(first_word=0, last_word=2, score=70.0, chosen=False),
        Take(first_word=5, last_word=10, score=100.0, chosen=True),
    ]
    alignment = _alignment(
        [ScriptSentence(id=0, chapter=None, text="A neuron is a weighted vote.", takes=takes)]
    )
    decisions = detect_retakes(words, LONG, alignment=alignment, script_config=SCRIPT)

    assert len(decisions) == 1
    assert decisions[0].dropped_words == (0, 4)
    assert decisions[0].kept_from_word == 5
    assert decisions[0].match_words == 3
    assert decisions[0].action == "drop"
    assert decisions[0].evidence == "script"

    narrow = LONG.model_copy(update={"window_words": 2})
    held = detect_retakes(words, narrow, alignment=alignment, script_config=SCRIPT)
    assert held[0].dropped_words == (0, 2)


def test_run_retakes_hashes_claps_that_exist_and_ignores_missing_files(tmp_path: Path):
    words = _indexed(
        _words(["the", "agent", "the", "agent", "joins"], starts=[0.0, 0.4, 1.2, 1.6, 2.0])
    )
    _write_words(tmp_path, words)
    words_path = tmp_path / "artifacts" / "words.json"
    first = run_retakes(tmp_path)
    assert first.data.decisions == []
    assert first.meta.inputs_hash == inputs_hash(artifacts=[words_path])
    assert first.meta.stage_version == 4
    second = run_retakes(tmp_path)
    assert second.meta.created_at == first.meta.created_at

    events_path = tmp_path / "artifacts" / "audio_events.json"
    write_artifact(
        events_path,
        AudioEventsArtifact(
            meta=make_meta(
                stage="audio",
                stage_version=1,
                inputs_hash="sha256:audio",
                config_hash="sha256:cfg",
            ),
            data=AudioEventsData(backend="none", speech=[], claps=[_clap("01", 0.9)]),
        ),
    )
    third = run_retakes(tmp_path)
    assert third.meta.inputs_hash == inputs_hash(artifacts=[words_path, events_path])
    assert third.meta.inputs_hash != first.meta.inputs_hash
    assert third.data.decisions[0].evidence == "clap"
    assert third.data.decisions[0].dropped_words == (0, 1)


def test_run_retakes_reads_alignment_when_the_file_exists(tmp_path: Path):
    words = _neuron_pair()
    _write_words(tmp_path, words)
    plain = run_retakes(tmp_path)
    assert plain.data.decisions[0].evidence == "transcript"

    alignment_path = tmp_path / "artifacts" / "alignment.json"
    write_artifact(
        alignment_path,
        AlignmentArtifact(
            meta=make_meta(
                stage="align",
                stage_version=1,
                inputs_hash="sha256:align",
                config_hash="sha256:cfg",
            ),
            data=_alignment(
                [
                    ScriptSentence(
                        id=0,
                        chapter=None,
                        text="A neuron is a weighted vote.",
                        takes=_neuron_takes(100.0, 95.0),
                    )
                ]
            ),
        ),
    )
    scripted = run_retakes(tmp_path)
    assert scripted.meta.inputs_hash != plain.meta.inputs_hash
    assert scripted.data.decisions[0].evidence == "script"
    assert scripted.data.decisions[0].dropped_words == (6, 11)
    assert scripted.data.decisions[0].flag is False


@st.composite
def _scenarios(draw: st.DrawFn):
    count = draw(st.integers(min_value=0, max_value=16))
    tokens = [draw(st.sampled_from(_TOKENS)) for _ in range(count)]
    words = _indexed(_words(tokens, step=0.5, duration=0.3)) if tokens else []
    claps: list[Clap] = []
    if words:
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            t = draw(
                st.floats(
                    min_value=0,
                    max_value=words[-1].end + 0.5,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            claps.append(Clap(source=words[0].source, t=t, peak_db=-3.0, rise_db=25.0))
    alignment = None
    if len(words) >= 6 and draw(st.booleans()):
        mid = len(words) // 2
        score = draw(
            st.floats(min_value=80, max_value=100, allow_nan=False, allow_infinity=False)
        )
        alignment = AlignmentData(
            script=ScriptInfo(path="script.md", hash="sha256:x"),
            chapters=[],
            sentences=[
                ScriptSentence(
                    id=0,
                    chapter=None,
                    text="line",
                    takes=[
                        Take(first_word=0, last_word=mid - 1, score=100.0, chosen=True),
                        Take(first_word=mid, last_word=len(words) - 1, score=score, chosen=False),
                    ],
                )
            ],
            unscripted=[],
            missing=[],
        )
    return words, claps, alignment


@settings(max_examples=40)
@given(_scenarios())
def test_dropped_spans_do_not_overlap(scenario) -> None:
    words, claps, alignment = scenario
    plain = detect_retakes(words, LONG)
    _assert_disjoint(plain)
    clapped = detect_retakes(words, LONG, claps=claps, claps_config=CLAPS)
    _assert_disjoint(clapped)
    both = detect_retakes(
        words,
        LONG,
        claps=claps,
        claps_config=CLAPS,
        alignment=alignment,
        script_config=SCRIPT,
    )
    _assert_disjoint(both)

"""Retake detection on synthetic word lists. Thresholds come from profile ``long``."""

from pathlib import Path

from typer.testing import CliRunner

from cutter.cli import app
from cutter.config import load_profile
from cutter.models import Word, WordsArtifact, WordsData, make_meta, write_artifact
from cutter.retakes import detect_retakes, run_retakes

LONG = load_profile("long").retakes


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


def test_simple_retake_drops_the_failed_take():
    # The em dash in the spec is a pause, not a word.
    words = _indexed(
        _words(["the", "agent", "joins", "the", "the", "agent", "joins", "the", "call"])
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.id == "d001"
    assert decision.kind == "retake"
    assert decision.dropped_words == (0, 3)
    assert decision.kept_from_word == 4
    assert decision.match_words == 4
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.flag_reason is None
    assert decision.judge is None


def test_short_aborted_take_drops_three_words():
    words = _indexed(_words(["so", "the", "agent", "so", "the", "agent", "joins", "the", "call"]))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].dropped_words == (0, 2)
    assert decisions[0].kept_from_word == 3
    assert decisions[0].action == "drop"
    assert decisions[0].flag is False


def test_chain_of_three_aborted_takes_keeps_the_last():
    words = _indexed(_words(["the", "agent", "joins"] * 3))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 2
    assert [decision.id for decision in decisions] == ["d001", "d002"]
    assert [decision.dropped_words for decision in decisions] == [(0, 2), (3, 5)]
    assert [decision.kept_from_word for decision in decisions] == [3, 6]
    assert all(decision.action == "drop" and decision.flag is False for decision in decisions)


def test_stopword_only_repeat_is_not_a_retake():
    # The ellipsis in the spec is a pause, not a word.
    words = _indexed(_words(["and", "the", "and", "the", "server"]))
    assert detect_retakes(words, LONG) == []


def test_repetition_outside_lookback_is_not_a_retake():
    phrase = ["the", "agent", "joins", "the", "call"]
    words = _indexed(_words(phrase, start=0.0), _words(phrase, start=120.0))
    assert detect_retakes(words, LONG) == []


def test_long_aborted_segment_is_kept_and_flagged():
    words = _indexed(
        _words(["the", "agent", "joins"], starts=[0.0, 0.4, 22.0]),
        _words(["the", "agent", "joins"], start=22.6),
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 2)
    assert decision.dropped_duration_s > LONG.auto_drop_max_s
    assert decision.action == "keep"
    assert decision.flag is True
    assert decision.flag_reason == "long_segment"


def test_exact_finished_sentence_is_dropped_once():
    sentence = ["A", "neuron", "is", "a", "weighted", "vote."]
    words = _indexed(_words(sentence + sentence))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 5)
    assert decision.kept_from_word == 6
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.flag_reason is None


def test_exact_opening_question_is_dropped_once():
    # The sample transcript says this question twice, about a second apart.
    question = [
        "Do",
        "you",
        "want",
        "your",
        "designs",
        "to",
        "go",
        "from",
        "this",
        "to",
        "this?",
    ]
    words = _indexed(
        _words(
            question,
            starts=[3.68, 3.92, 4.08, 4.24, 4.4, 4.88, 5.12, 5.36, 5.68, 6.0, 6.32],
        ),
        _words(
            question,
            starts=[8.24, 8.48, 8.64, 8.8, 9.12, 9.84, 10.16, 10.48, 10.8, 11.12, 11.76],
        ),
        _words(["Then", "you", "should"], start=13.12),
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 10)
    assert decision.kept_from_word == 11
    assert decision.match_words == 11
    assert decision.action == "drop"
    assert decision.flag is False


def test_different_finished_sentence_stays_once():
    first = ["A", "neuron", "is", "a", "weighted", "vote."]
    second = ["A", "neuron", "is", "a", "weighted", "guess."]
    words = _indexed(_words(first + second))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 5)
    assert decision.kept_from_word == 6
    assert decision.action == "keep"
    assert decision.flag is True
    assert decision.flag_reason == "complete_sentence"


def test_cross_file_retake_drops_the_aborted_tail():
    words = _indexed(
        _words(["welcome", "back", "the", "agent", "joins", "the"], source="01"),
        _words(["the", "agent", "joins", "the", "call"], source="02"),
    )
    decisions = detect_retakes(words, LONG)

    assert words[5].source == "01"
    assert words[6].source == "02"
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (2, 5)
    assert decision.kept_from_word == 6
    assert decision.action == "drop"
    assert decision.flag is False


def test_exact_finished_sentence_across_files_is_dropped():
    sentence = ["A", "neuron", "is", "a", "weighted", "vote."]
    words = _indexed(_words(sentence, source="01"), _words(sentence, source="02"))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 5)
    assert decision.kept_from_word == 6
    assert decision.action == "drop"
    assert decision.flag is False
    assert decision.flag_reason is None


def test_one_mismatch_inside_the_window_still_matches():
    words = _indexed(_words(["the", "cat", "joins", "the", "the", "dog", "joins", "the", "call"]))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].dropped_words == (0, 3)
    assert decisions[0].match_words == 3
    assert decisions[0].action == "drop"


def test_missing_content_flags_words_the_later_take_lacks():
    words = _indexed(
        _words(
            [
                "the",
                "agent",
                "joins",
                "server",
                "cluster",
                "database",
                "the",
                "agent",
                "joins",
                "the",
                "call",
            ]
        )
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].action == "drop"
    assert decisions[0].flag is True
    assert decisions[0].flag_reason == "retake_missing_content"


def test_repeated_content_word_uses_set_difference():
    # "server" is said three times in the failed take and once in the later
    # take. As a set it is not missing, so the guard does not flag.
    words = _indexed(
        _words(
            [
                "the",
                "agent",
                "joins",
                "server",
                "server",
                "server",
                "the",
                "agent",
                "joins",
                "the",
                "server",
                "call",
            ]
        )
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].action == "drop"
    assert decisions[0].flag is False
    assert decisions[0].flag_reason is None


def test_empty_content_set_skips_the_missing_content_guard():
    stops = ["in", "on", "for", "with", "at"]
    words = _indexed(_words(stops + stops + ["server"]))
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].action == "drop"
    assert decisions[0].flag is False
    assert decisions[0].flag_reason is None


def test_long_exact_finished_sentence_is_dropped():
    sentence = ["A", "neuron", "is", "a", "weighted", "vote."]
    words = _indexed(
        _words(sentence, starts=[0.0, 0.4, 0.8, 1.2, 1.6, 22.0]),
        _words(sentence, start=23.0),
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    assert decisions[0].dropped_duration_s > LONG.auto_drop_max_s
    assert decisions[0].action == "drop"
    assert decisions[0].flag is False


def test_different_finished_sentence_beats_a_long_segment():
    first = ["A", "neuron", "is", "a", "weighted", "vote."]
    second = ["A", "neuron", "is", "a", "weighted", "guess."]
    words = _indexed(
        _words(first, starts=[0.0, 0.4, 0.8, 1.2, 1.6, 22.0]),
        _words(second, start=23.0),
    )
    decisions = detect_retakes(words, LONG)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_duration_s > LONG.auto_drop_max_s
    assert decision.action == "keep"
    assert decision.flag_reason == "complete_sentence"


def test_stopwords_file_replaces_the_builtin_list(tmp_path: Path):
    path = tmp_path / "stops.txt"
    path.write_text("the\n\nagent\njoins\ncall\n", encoding="utf-8")
    config = LONG.model_copy(update={"stopwords_file": str(path)})
    words = _indexed(
        _words(["the", "agent", "joins", "the", "the", "agent", "joins", "the", "call"])
    )
    assert detect_retakes(words, config) == []


def test_run_retakes_writes_decisions_and_uses_the_cache(tmp_path: Path):
    words = _indexed(
        _words(["the", "agent", "joins", "the", "the", "agent", "joins", "the", "call"])
    )
    write_artifact(
        tmp_path / "artifacts" / "words.json",
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
    first = run_retakes(tmp_path)
    second = run_retakes(tmp_path)

    assert first.data.decisions[0].dropped_words == (0, 3)
    assert second.meta.created_at == first.meta.created_at
    assert (tmp_path / "artifacts" / "decisions.json").is_file()


def test_cli_retakes_reports_the_drop(tmp_path: Path):
    words = _indexed(
        _words(["the", "agent", "joins", "the", "the", "agent", "joins", "the", "call"])
    )
    write_artifact(
        tmp_path / "artifacts" / "words.json",
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
    result = CliRunner().invoke(app, ["retakes", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 decisions, 1 dropped, 0 flagged" in result.stdout


def test_cli_retakes_without_words_exits_1(tmp_path: Path):
    result = CliRunner().invoke(app, ["retakes", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing words artifact" in result.output

"""Script alignment on synthetic word lists. No audio and no network."""

import hashlib
from datetime import UTC, datetime
from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from cutter.align import STAGE_VERSION, align_script, run_align
from cutter.config import load_profile
from cutter.models import (
    AlignmentData,
    Take,
    Word,
    WordsArtifact,
    WordsData,
    WordSpan,
    make_meta,
    write_artifact,
)
from cutter.script import script_bytes_hash
from cutter.transcribe import normalize_word

LONG = load_profile("long")
_SIMILARITY = LONG.retakes.word_similarity
_MIN_SCORE = LONG.script.min_take_score

_LINE = (
    "The cache stores the result and skips the work.",
    "The agent joins the call and greets the room.",
    "The host ends the session after the demo.",
    "The guest waves once and leaves the stage.",
)
_WRONG = "The agent joins the call and greets the roam."
_VOCAB = (
    "crimson",
    "lantern",
    "marble",
    "orchid",
    "pebble",
    "quartz",
    "sandal",
    "timber",
    "velvet",
    "walnut",
    "yellow",
    "zircon",
)


def test_three_sentences_spoken_once():
    data = _align(_LINE[:3], _LINE[:3])
    chosen = _chosen(data)
    assert len(chosen) == 3
    assert data.missing == []
    assert data.unscripted == []
    assert [sentence.chapter for sentence in data.sentences] == ["c01", "c02", "c03"]


def test_worse_second_take_is_an_alternate():
    spoken = [_LINE[0], _LINE[1], _WRONG, _LINE[2]]
    data = _align(_LINE[:3], spoken)
    takes = data.sentences[1].takes
    assert len(takes) == 2
    assert takes[0].chosen is True
    assert takes[1].chosen is False
    assert takes[0].first_word < takes[1].first_word
    assert takes[1].score < takes[0].score
    assert takes[1].score >= _MIN_SCORE
    assert data.missing == []
    _chosen(data)


def test_identical_second_sentence_chooses_the_later_take():
    spoken = [_LINE[0], _LINE[1], _LINE[1], _LINE[2]]
    data = _align(_LINE[:3], spoken)
    takes = data.sentences[1].takes
    assert len(takes) == 2
    assert takes[0].chosen is False
    assert takes[1].chosen is True
    assert takes[0].score == takes[1].score
    assert takes[0].first_word < takes[1].first_word
    _chosen(data)


def test_prefer_last_chooses_the_later_take_when_the_first_is_better():
    spoken = [_LINE[0], _LINE[1], _WRONG, _LINE[2]]
    data = _align(_LINE[:3], spoken, prefer="last")
    takes = data.sentences[1].takes
    assert takes[0].score > takes[1].score
    assert takes[0].chosen is False
    assert takes[1].chosen is True
    _chosen(data)


def test_aside_between_sentences_is_one_unscripted_span():
    aside = [f"aside{index:02d}" for index in range(15)]
    first = _tokens(_LINE[0])
    second = _tokens(_LINE[1])
    data = _align(_LINE[:2], [" ".join(first), " ".join(aside), " ".join(second)])
    assert len(_chosen(data)) == 2
    assert data.missing == []
    assert data.unscripted == [WordSpan(first_word=len(first), last_word=len(first) + 14)]


def test_out_of_order_match_stays_a_take():
    """A matched sentence the monotonic pass cannot place is not unspoken."""
    first = "The cache stores the result."
    second = "The agent joins the call."
    data = _align([first, second], [second, first])
    skipped = data.sentences[1]
    assert data.missing == []
    assert skipped.takes
    assert not any(take.chosen for take in skipped.takes)
    assert data.unscripted == []
    assert len(_chosen(data)) == 1


def test_unspoken_sentence_is_missing():
    data = _align(_LINE, _LINE[:3])
    assert data.missing == [3]
    assert data.sentences[3].takes == []
    assert len(_chosen(data)) == 3


def test_sentence_across_a_source_boundary_is_missing():
    tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    words = _join(
        _words(tokens[:3], source="s01"),
        _words(tokens[3:], source="s02"),
    )
    data = align_script(
        words,
        " ".join(tokens) + ".",
        levels=[1, 2],
        word_similarity=_SIMILARITY,
        min_take_score=_MIN_SCORE,
        prefer="best",
        path="script.md",
        script_hash="sha256:test",
    )
    assert data.missing == [0]
    assert data.sentences[0].takes == []
    assert data.unscripted == [WordSpan(first_word=0, last_word=len(tokens) - 1)]


def test_word_similarity_threshold_is_passed_in():
    words = _words(["helo", "world"])
    loose = align_script(
        words,
        "hello world.",
        levels=[1],
        word_similarity=80,
        min_take_score=_MIN_SCORE,
        prefer="best",
        path="script.md",
        script_hash="sha256:test",
    )
    strict = align_script(
        words,
        "hello world.",
        levels=[1],
        word_similarity=95,
        min_take_score=_MIN_SCORE,
        prefer="best",
        path="script.md",
        script_hash="sha256:test",
    )
    assert loose.missing == []
    assert strict.missing == [0]
    assert strict.sentences[0].takes == []


def test_run_align_stores_the_script_hash(tmp_path):
    _write_words(tmp_path, ["Hello", "world."])
    raw = b"# Intro\n\nHello world.\n"
    (tmp_path / "script.md").write_bytes(raw)
    first = run_align(tmp_path, LONG)
    assert first.meta.stage_version == STAGE_VERSION
    assert first.data.script is not None
    assert first.data.script.path == "script.md"
    assert first.data.script.hash == script_bytes_hash(raw)
    assert first.data.script.hash == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert first.data.sentences[0].takes[0].chosen is True
    second = run_align(tmp_path, LONG)
    assert second.meta.created_at == first.meta.created_at


def _align(
    script_lines: tuple[str, ...] | list[str],
    spoken_lines: tuple[str, ...] | list[str],
    *,
    prefer: Literal["best", "last"] = "best",
) -> AlignmentData:
    spoken: list[str] = []
    for line in spoken_lines:
        spoken.extend(_tokens(line))
    parts = [f"# Part {index}\n\n{line}\n" for index, line in enumerate(script_lines, start=1)]
    return align_script(
        _words(spoken),
        "\n".join(parts),
        levels=[1, 2],
        word_similarity=_SIMILARITY,
        min_take_score=_MIN_SCORE,
        prefer=prefer,
        path="script.md",
        script_hash="sha256:test",
    )


def _chosen(data: AlignmentData) -> list[Take]:
    picked: list[Take] = []
    for sentence in data.sentences:
        chosen = [take for take in sentence.takes if take.chosen]
        if sentence.id in data.missing:
            assert sentence.takes == []
            assert chosen == []
        elif chosen:
            assert len(chosen) == 1
            picked.append(chosen[0])
        else:
            assert sentence.takes
        assert sentence.takes == sorted(sentence.takes, key=lambda take: take.first_word)
    for previous, nxt in zip(picked, picked[1:], strict=False):
        assert nxt.first_word > previous.first_word
        assert nxt.first_word > previous.last_word
    return picked


def _tokens(line: str) -> list[str]:
    return [normalize_word(token) for token in line.split() if normalize_word(token)]


def _words(tokens: list[str], *, source: str = "s01") -> list[Word]:
    words: list[Word] = []
    for offset, token in enumerate(tokens):
        words.append(
            Word(
                i=offset,
                source=source,
                w=token,
                norm=normalize_word(token),
                start=offset * 0.4,
                end=offset * 0.4 + 0.3,
                sent=0,
            )
        )
    return words


def _join(*groups: list[Word]) -> list[Word]:
    words: list[Word] = []
    for group in groups:
        for word in group:
            words.append(word.model_copy(update={"i": len(words)}))
    return words


def _write_words(project, tokens: list[str]) -> None:
    write_artifact(
        project / "artifacts" / "words.json",
        WordsArtifact(
            meta=make_meta(
                stage="transcribe",
                stage_version=1,
                inputs_hash="sha256:in",
                config_hash="sha256:cfg",
                created_at=datetime(2026, 10, 8, tzinfo=UTC),
            ),
            data=WordsData(words=_words(tokens)),
        ),
    )


@st.composite
def _matchable(draw: st.DrawFn):
    count = draw(st.integers(min_value=1, max_value=3))
    cursor = 0
    sentences: list[list[str]] = []
    spoken: list[str] = []
    for _ in range(count):
        length = draw(st.integers(min_value=2, max_value=4))
        sentence = list(_VOCAB[cursor : cursor + length])
        cursor += length
        sentences.append(sentence)
        for _repeat in range(draw(st.integers(min_value=1, max_value=2))):
            spoken.extend(sentence)
        for gap in range(draw(st.integers(min_value=0, max_value=2))):
            spoken.append(f"qq{cursor}{gap}")
    return sentences, spoken


def _matchable_markdown(sentences: list[list[str]]) -> str:
    lines = ["---", "title: draft", "---", ""]
    for index, sentence in enumerate(sentences, start=1):
        lines.append(f"# Take {index}")
        lines.append("")
        lines.append(" ".join(sentence) + ".")
        lines.append("")
    return "\n".join(lines)


@settings(max_examples=20, deadline=2000)
@given(_matchable())
def test_chosen_takes_are_strictly_increasing_and_non_overlapping(case):
    sentences, spoken = case
    markdown = _matchable_markdown(sentences)
    data = align_script(
        _words(spoken),
        markdown,
        levels=[1, 2],
        word_similarity=_SIMILARITY,
        min_take_score=_MIN_SCORE,
        prefer="best",
        path="script.md",
        script_hash="sha256:test",
    )
    chosen = _chosen(data)
    assert data.missing == []
    assert len(chosen) == len(sentences)

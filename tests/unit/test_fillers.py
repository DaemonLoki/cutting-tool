"""Filler decisions on synthetic word lists. Thresholds come from profile ``long``."""

from pathlib import Path
from typing import Literal

import pytest

from cutter.config import load_profile
from cutter.fillers import STAGE_VERSION, detect_fillers, run_fillers
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)

FILLERS = load_profile("long").fillers
PROFILE = load_profile("long")


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


def _span_norms(words: list[Word], decision: Decision) -> list[str]:
    first, last = decision.dropped_words
    return [word.norm for word in words if first <= word.i <= last]


def _retake(
    first: int,
    last: int,
    *,
    action: Literal["drop", "keep"] = "drop",
) -> Decision:
    return Decision(
        id="d001",
        kind="retake",
        dropped_words=(first, last),
        kept_from_word=last + 1,
        match_words=1,
        dropped_duration_s=0.3,
        action=action,
        flag=action == "keep",
        flag_reason="long_segment" if action == "keep" else None,
    )


def test_um_and_uh_drop_and_so_stays_unless_it_opens_the_sentence():
    words = _indexed(_words(["Um,", "so", "the", "agent,", "uh,", "joins", "the", "call."]))
    decisions = detect_fillers(words, [], FILLERS)

    assert [decision.id for decision in decisions] == ["f001", "f002"]
    assert [_span_norms(words, decision) for decision in decisions] == [["um"], ["uh"]]
    assert all(decision.action == "drop" and decision.flag is False for decision in decisions)
    assert all(decision.flag_reason is None for decision in decisions)
    um = words[0]
    uh = words[4]
    assert decisions[0].dropped_words == (um.i, um.i)
    assert decisions[0].kept_from_word == um.i + 1
    assert decisions[0].dropped_duration_s == um.end - um.start
    assert decisions[1].dropped_words == (uh.i, uh.i)
    assert decisions[1].kept_from_word == uh.i + 1
    for decision in decisions:
        assert decision.kind == "filler"
        assert decision.evidence == "transcript"
        assert decision.match_words == 0
        assert decision.judge is None
        assert decision.clap_s is None
        assert decision.score_gap is None

    listed = FILLERS.model_copy(update={"sentence_start_words": ["so"]})
    with_so = detect_fillers(words, [], listed)
    assert [_span_norms(words, decision) for decision in with_so] == [["um"], ["so"], ["uh"]]
    assert [decision.action for decision in with_so] == ["drop", "drop", "drop"]

    mid = _indexed(_words(["The", "agent,", "so,", "joins", "the", "call."]))
    assert detect_fillers(mid, [], listed) == []

    opening = _indexed(_words(["So", "the", "agent", "joins", "the", "call."]))
    assert detect_fillers(opening, [], FILLERS) == []
    opened = detect_fillers(opening, [], listed)
    assert [_span_norms(opening, decision) for decision in opened] == [["so"]]

    anywhere = FILLERS.model_copy(
        update={"words": ["so", *FILLERS.words], "sentence_start_words": ["so"]}
    )
    dropped = detect_fillers(mid, [], anywhere)
    assert [_span_norms(mid, decision) for decision in dropped] == [["so"]]


def test_consecutive_fillers_are_one_decision():
    words = _indexed(_words(["Um,", "uh,", "joins", "the", "call."]))
    decisions = detect_fillers(words, [], FILLERS)

    assert len(decisions) == 1
    assert decisions[0].id == "f001"
    assert decisions[0].dropped_words == (0, 1)
    assert _span_norms(words, decisions[0]) == ["um", "uh"]
    assert decisions[0].kept_from_word == 2
    assert decisions[0].action == "drop"


def test_you_know_drops_only_when_the_phrase_is_listed():
    words = _indexed(_words(["you", "know", "the", "agent."]))
    assert detect_fillers(words, [], FILLERS) == []

    listed = FILLERS.model_copy(update={"phrases": ["you know"]})
    decisions = detect_fillers(words, [], listed)
    assert len(decisions) == 1
    assert decisions[0].dropped_words == (0, 1)
    assert _span_norms(words, decisions[0]) == ["you", "know"]
    assert decisions[0].action == "drop"
    assert decisions[0].match_words == 0

    both = FILLERS.model_copy(update={"words": ["you", *FILLERS.words], "phrases": ["you know"]})
    phrase = detect_fillers(words, [], both)
    assert len(phrase) == 1
    assert phrase[0].dropped_words == (0, 1)


def test_filler_inside_a_dropped_retake_is_not_a_decision():
    words = _indexed(_words(["Um,", "the", "agent", "joins", "the", "call."]))
    assert detect_fillers(words, [_retake(0, 0)], FILLERS) == []

    kept = detect_fillers(words, [_retake(0, 0, action="keep")], FILLERS)
    assert len(kept) == 1
    assert kept[0].dropped_words == (0, 0)
    assert kept[0].action == "drop"


def test_dropped_words_between_fillers_stay_one_run():
    words = _indexed(_words(["Um,", "nope,", "uh,", "joins", "the", "call."]))
    decisions = detect_fillers(words, [_retake(1, 1)], FILLERS)

    assert len(decisions) == 1
    assert decisions[0].dropped_words == (0, 2)
    assert decisions[0].action == "drop"


def test_long_filler_is_kept_and_flagged():
    words = _indexed(_words(["Um,", "hello."], starts=[0.0, 2.2], duration=2.0))
    decisions = detect_fillers(words, [], FILLERS)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.dropped_words == (0, 0)
    assert decision.dropped_duration_s == words[0].end - words[0].start
    assert decision.dropped_duration_s > FILLERS.max_duration_s
    assert decision.action == "keep"
    assert decision.flag is True
    assert decision.flag_reason == "filler_long"
    assert decision.kind == "filler"
    assert decision.evidence == "transcript"


def test_hmm_alone_in_its_sentence_is_content():
    words = _indexed(_words(["Hmm."]), _words(["The", "agent", "joins."]))
    assert detect_fillers(words, [], FILLERS) == []

    followed = _indexed(_words(["Hmm,", "the", "agent", "joins."]))
    decisions = detect_fillers(followed, [], FILLERS)
    assert len(decisions) == 1
    assert _span_norms(followed, decisions[0]) == ["hmm"]
    assert decisions[0].action == "drop"


def test_disabled_fillers_write_no_decisions(tmp_path: Path):
    words = _indexed(_words(["Um,", "the", "agent", "joins."]))
    disabled = PROFILE.model_copy(update={"fillers": FILLERS.model_copy(update={"enabled": False})})
    assert detect_fillers(words, [], disabled.fillers) == []

    _write_words(tmp_path, words)
    _write_decisions(tmp_path, [])
    artifact = run_fillers(tmp_path, disabled)
    assert artifact.meta.stage == "fillers"
    assert artifact.meta.stage_version == STAGE_VERSION
    assert artifact.data.decisions == []


def test_cache_hits_until_filler_words_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    words = _indexed(_words(["Um,", "the", "agent", "joins."]))
    _write_words(tmp_path, words)
    _write_decisions(tmp_path, [])
    first = run_fillers(tmp_path, PROFILE)

    assert first.meta.stage == "fillers"
    assert first.meta.stage_version == STAGE_VERSION
    assert len(first.data.decisions) == 1
    assert first.data.decisions[0].action == "drop"

    def _unexpected_meta(**_kwargs: object) -> None:
        raise AssertionError("cache miss")

    monkeypatch.setattr("cutter.fillers.make_meta", _unexpected_meta)
    second = run_fillers(tmp_path, PROFILE)
    assert second.meta.created_at == first.meta.created_at
    monkeypatch.undo()

    changed = PROFILE.model_copy(update={"fillers": FILLERS.model_copy(update={"words": ["uh"]})})
    third = run_fillers(tmp_path, changed)
    assert third.data.decisions == []
    assert third.meta.inputs_hash == first.meta.inputs_hash
    assert third.meta.config_hash != first.meta.config_hash

    _write_decisions(tmp_path, [_retake(0, 0)])
    fourth = run_fillers(tmp_path, PROFILE)
    assert fourth.data.decisions == []
    assert fourth.meta.inputs_hash != first.meta.inputs_hash


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


def _write_decisions(project: Path, decisions: list[Decision]) -> None:
    write_artifact(
        project / "artifacts" / "decisions.json",
        DecisionsArtifact(
            meta=make_meta(
                stage="judge",
                stage_version=1,
                inputs_hash="sha256:in",
                config_hash="sha256:cfg",
            ),
            data=DecisionsData(decisions=decisions),
        ),
    )

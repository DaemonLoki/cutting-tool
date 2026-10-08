"""Judge verdicts on synthetic word lists. Thresholds come from profile ``long``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from tests.unit.test_retakes_phase2 import _alignment, _neuron_pair, _neuron_takes

from cutter.config import load_profile
from cutter.judge import STAGE_VERSION, judge_decisions, run_judge
from cutter.llm import EndpointUnreachable
from cutter.models import (
    AlignmentArtifact,
    Decision,
    DecisionsArtifact,
    ScriptSentence,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.retakes import detect_retakes, run_retakes

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


def _long_words() -> list[Word]:
    return _indexed(
        _words(["the", "agent", "joins"], starts=[0.0, 0.4, 22.0]),
        _words(["the", "agent", "joins"], start=22.6),
    )


def _missing_words() -> list[Word]:
    return _indexed(
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


def _finished_words() -> list[Word]:
    first = ["A", "neuron", "is", "a", "weighted", "vote."]
    second = ["A", "neuron", "is", "a", "weighted", "guess."]
    return _indexed(_words(first + second))


def _one(words: list[Word]) -> Decision:
    decisions = detect_retakes(words, PROFILE.retakes)
    assert len(decisions) == 1
    return decisions[0]


class _Client:
    def __init__(self, verdict: object) -> None:
        self._verdict = verdict
        self.calls: list[tuple[str, str, dict]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict | None:
        self.calls.append((system, user, schema))
        if isinstance(self._verdict, Exception):
            raise self._verdict
        if isinstance(self._verdict, list):
            return self._verdict.pop(0)
        return self._verdict


def _verdict(choice: str, confidence: float, reason: str = "because") -> dict[str, object]:
    return {"choice": choice, "confidence": confidence, "reason": reason}


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


def _judge(decision: Decision, words: list[Word], verdict: object) -> tuple[Decision, _Client]:
    client = _Client(verdict)
    judged = judge_decisions([decision], words, PROFILE.judge, client)
    assert len(judged) == 1
    assert decision.judge is None
    return judged[0], client


def test_choice_b_drops_and_unflags() -> None:
    words = _long_words()
    decision = _one(words)
    assert decision.flag_reason == "long_segment"
    judged, client = _judge(decision, words, _verdict("B", 0.9))

    assert judged.action == "drop"
    assert judged.flag is False
    assert judged.flag_reason is None
    assert judged.judge is not None
    assert judged.judge.choice == "B"
    assert judged.judge.confidence == 0.9
    assert judged.judge.model == PROFILE.judge.model
    assert decision.action == "keep"
    assert decision.flag is True
    schema = client.calls[0][2]
    assert schema["required"] == ["choice", "confidence", "reason"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["choice"]["enum"] == ["A", "B", "both"]
    assert schema["properties"]["reason"]["maxLength"] == 300


def test_choice_both_keeps_and_unflags() -> None:
    words = _long_words()
    judged, _client = _judge(_one(words), words, _verdict("both", 0.9))

    assert judged.action == "keep"
    assert judged.flag is False
    assert judged.flag_reason is None
    assert judged.judge is not None
    assert judged.judge.choice == "both"


def test_choice_a_keeps_the_first_take_flagged() -> None:
    words = _long_words()
    judged, _client = _judge(_one(words), words, _verdict("A", 0.9))

    assert judged.action == "keep"
    assert judged.flag is True
    assert judged.flag_reason == "judge_prefers_first_take"
    assert judged.judge is not None
    assert judged.judge.choice == "A"


def test_low_confidence_stays_flagged_with_the_original_reason() -> None:
    assert PROFILE.judge.min_confidence > 0
    words = _missing_words()
    decision = _one(words)
    assert decision.action == "drop"
    assert decision.flag_reason == "retake_missing_content"
    low = PROFILE.judge.min_confidence / 2
    judged, _client = _judge(decision, words, _verdict("B", low, "unsure"))

    assert judged.action == "keep"
    assert judged.flag is True
    assert judged.flag_reason == "retake_missing_content"
    assert judged.judge is not None
    assert judged.judge.choice == "B"
    assert judged.judge.confidence == low
    assert judged.judge.model == PROFILE.judge.model


def test_confidence_at_the_minimum_applies_the_choice() -> None:
    words = _long_words()
    judged, _client = _judge(_one(words), words, _verdict("B", PROFILE.judge.min_confidence))

    assert judged.action == "drop"
    assert judged.flag is False


def test_complete_sentence_is_not_sent() -> None:
    words = _finished_words()
    decisions = detect_retakes(words, PROFILE.retakes)
    assert decisions
    client = _Client(_verdict("B", 0.9))
    judged = judge_decisions(decisions, words, PROFILE.judge, client)

    assert client.calls == []
    assert all(
        decision.action == "keep"
        and decision.flag is True
        and decision.flag_reason == "complete_sentence"
        and decision.judge is None
        for decision in judged
    )


def test_long_segment_is_sent_and_a_plain_drop_is_not() -> None:
    words = _long_words()
    long_segment = _one(words)
    missing = long_segment.model_copy(
        update={"id": "d002", "action": "drop", "flag_reason": "retake_missing_content"}
    )
    plain = long_segment.model_copy(
        update={"id": "d003", "action": "drop", "flag": False, "flag_reason": None}
    )
    client = _Client([_verdict("B", 0.9), _verdict("B", 0.9)])
    judged = judge_decisions([plain, long_segment, missing], words, PROFILE.judge, client)

    assert len(client.calls) == 2
    assert judged[0] == plain
    assert judged[1].action == "drop" and judged[1].flag is False
    assert judged[2].action == "drop" and judged[2].flag is False
    assert judged[2].judge is not None
    assert judged[2].judge.choice == "B"


def test_invalid_verdict_leaves_the_decision_unchanged() -> None:
    words = _long_words()
    decision = _one(words)
    payloads: list[object] = [
        None,
        {"confidence": 0.9, "reason": "missing choice"},
        {"choice": "C", "confidence": 0.9, "reason": "no"},
        {"choice": "B", "confidence": 1.5, "reason": "high"},
        {"choice": "B", "confidence": -0.1, "reason": "low"},
        {"choice": "B", "confidence": True, "reason": "bool"},
        {"choice": "B", "confidence": 0.9, "reason": "x" * 301},
    ]
    for payload in payloads:
        judged, client = _judge(decision, words, payload)
        assert client.calls
        assert judged == decision
        assert judged.judge is None


def test_unreachable_endpoint_leaves_every_decision(caplog: pytest.LogCaptureFixture) -> None:
    words = _long_words()
    first = _one(words)
    second = first.model_copy(
        update={"id": "d002", "action": "drop", "flag_reason": "retake_missing_content"}
    )
    client = _Client(EndpointUnreachable("down"))
    with caplog.at_level(logging.WARNING, logger="cutter.judge"):
        judged = judge_decisions([first, second], words, PROFILE.judge, client)

    assert client.calls and len(client.calls) == 1
    assert judged == [first, second]
    assert judged[0] is not first
    messages = [record.message for record in caplog.records if record.name == "cutter.judge"]
    assert messages == ["judge endpoint unreachable; leaving decisions unchanged"]


def test_run_judge_unreachable_keeps_the_retakes_artifact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    words = _long_words()
    _write_words(tmp_path, words)
    retakes = run_retakes(tmp_path)
    assert retakes.meta.stage == "retakes"
    path = tmp_path / "artifacts" / "decisions.json"
    before = path.read_text(encoding="utf-8")
    client = _Client(EndpointUnreachable("down"))

    with caplog.at_level(logging.WARNING, logger="cutter.judge"):
        result = run_judge(tmp_path, client=client)

    assert len(client.calls) == 1
    assert result.meta.stage == "retakes"
    assert result.meta.created_at == retakes.meta.created_at
    assert path.read_text(encoding="utf-8") == before
    assert DecisionsArtifact.model_validate_json(before).meta.stage == "retakes"
    warnings = [record for record in caplog.records if record.name == "cutter.judge"]
    assert len(warnings) == 1


def test_no_llm_cache_does_not_satisfy_a_later_llm_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = _long_words()
    _write_words(tmp_path, words)
    client = _Client(_verdict("B", 0.9, "cut it"))
    first = run_judge(tmp_path, no_llm=True, client=client)

    assert client.calls == []
    assert first.meta.stage == "judge"
    assert first.meta.stage_version == STAGE_VERSION
    assert first.data.decisions == detect_retakes(words, PROFILE.retakes)

    def _unexpected_meta(**_kwargs: object) -> None:
        raise AssertionError("cache miss")

    monkeypatch.setattr("cutter.judge.make_meta", _unexpected_meta)
    second = run_judge(tmp_path, no_llm=True, client=client)
    assert second.meta.created_at == first.meta.created_at
    assert client.calls == []
    monkeypatch.undo()

    third = run_judge(tmp_path, no_llm=False, client=client)
    assert len(client.calls) == 1
    assert third.meta.stage == "judge"
    assert third.meta.inputs_hash != first.meta.inputs_hash
    assert third.meta.config_hash == first.meta.config_hash
    assert third.data.decisions[0].action == "drop"
    assert third.data.decisions[0].flag is False


def test_no_llm_judge_keeps_a_script_decision(tmp_path: Path) -> None:
    words = _neuron_pair()
    _write_words(tmp_path, words)
    plain = run_judge(tmp_path, no_llm=True)
    assert plain.data.decisions[0].evidence == "transcript"

    write_artifact(
        tmp_path / "artifacts" / "alignment.json",
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
    scripted = run_judge(tmp_path, no_llm=True)
    assert scripted.meta.inputs_hash != plain.meta.inputs_hash
    assert scripted.data.decisions[0].evidence == "script"
    assert scripted.data.decisions[0].dropped_words == (6, 11)
    assert scripted.data.decisions[0].flag is False


def test_judge_cache_misses_when_a_retakes_input_section_changes(tmp_path: Path) -> None:
    """Retakes reads ``claps`` and ``script`` too; a change there must rerun both stages."""
    words = _long_words()
    _write_words(tmp_path, words)
    client = _Client(_verdict("B", 0.9, "cut it"))
    first = run_judge(tmp_path, no_llm=True, client=client)

    changed = PROFILE.model_copy(
        update={"claps": PROFILE.claps.model_copy(update={"min_match_words": 4})}
    )
    second = run_judge(tmp_path, changed, no_llm=True, client=client)

    assert second.meta.inputs_hash != first.meta.inputs_hash
    assert second.meta.config_hash == first.meta.config_hash


def test_disabled_judge_uses_the_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = _long_words()
    _write_words(tmp_path, words)
    disabled = PROFILE.model_copy(
        update={"judge": PROFILE.judge.model_copy(update={"enabled": False})}
    )
    client = _Client(_verdict("B", 0.9))
    first = run_judge(tmp_path, disabled, client=client)

    assert client.calls == []
    assert first.meta.stage == "judge"
    assert first.data.decisions == detect_retakes(words, disabled.retakes)

    def _unexpected_meta(**_kwargs: object) -> None:
        raise AssertionError("cache miss")

    monkeypatch.setattr("cutter.judge.make_meta", _unexpected_meta)
    second = run_judge(tmp_path, disabled, client=client)
    assert second.meta.created_at == first.meta.created_at
    assert client.calls == []
    monkeypatch.undo()

    third = run_judge(tmp_path, PROFILE, client=client)
    assert len(client.calls) == 1
    assert third.meta.inputs_hash == first.meta.inputs_hash
    assert third.meta.config_hash != first.meta.config_hash
    assert third.data.decisions[0].action == "drop"


def test_run_judge_requires_words(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing words artifact"):
        run_judge(tmp_path, no_llm=True)


def test_run_judge_constructs_the_client_from_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _Fake:
        def __init__(self, *, base_url: str, model: str, timeout_s: float) -> None:
            captured["base_url"] = base_url
            captured["model"] = model
            captured["timeout_s"] = timeout_s

        def complete_json(self, system: str, user: str, schema: dict) -> dict[str, object]:
            return _verdict("B", 0.9, "cut the first")

    monkeypatch.setattr("cutter.judge.OpenAIJsonClient", _Fake)
    _write_words(tmp_path, _long_words())
    artifact = run_judge(tmp_path)

    assert captured == {
        "base_url": PROFILE.judge.base_url,
        "model": PROFILE.judge.model,
        "timeout_s": float(PROFILE.judge.timeout_s),
    }
    assert artifact.data.decisions[0].action == "drop"
    assert artifact.data.decisions[0].judge is not None
    assert artifact.data.decisions[0].judge.model == PROFILE.judge.model


def test_default_system_prompt_explains_the_choices() -> None:
    words = _long_words()
    _judged, client = _judge(_one(words), words, _verdict("both", 0.9))
    system = client.calls[0][0]

    assert "Segment A is a candidate failed attempt" in system
    assert "Segment B is the candidate retake" in system
    assert 'Answer "B" if A should be dropped' in system
    assert 'Answer "A" if B is the failed one and A should be kept instead' in system
    assert 'Answer "both" if this is not a retake' in system


def test_system_prompt_uses_the_skill_file_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("custom skill\n", encoding="utf-8")
    monkeypatch.setattr("cutter.judge._SKILL_PATH", skill)
    words = _long_words()
    _judged, client = _judge(_one(words), words, _verdict("both", 0.9))

    assert client.calls[0][0] == "custom skill\n"


def test_user_prompt_uses_nearby_sentences_without_times() -> None:
    groups = [
        ["One", "sentence", "here."],
        ["Two", "sentence", "here."],
        ["Three", "sentence", "here."],
        ["the", "agent", "joins"],
        ["the", "agent", "joins", "the", "call."],
        ["After", "the", "take."],
        ["Final", "sentence", "here."],
    ]
    words = _indexed(*[_words(group) for group in groups])
    first = sum(len(group) for group in groups[:3])
    decision = _one(_long_words()).model_copy(
        update={"dropped_words": (first, first + 2), "kept_from_word": first + 3}
    )
    _judged, client = _judge(decision, words, _verdict("both", 0.9))

    assert client.calls[0][1] == "\n".join(
        [
            "Before:",
            "Two sentence here. Three sentence here.",
            "",
            "A:",
            "the agent joins",
            "",
            "B:",
            "the agent joins the call. After the take.",
            "",
            "After:",
            "Final sentence here.",
        ]
    )
    assert "22.0" not in client.calls[0][1]


def test_user_prompt_includes_the_sentence_after_a_finished_span() -> None:
    groups = [
        ["Keep", "this."],
        ["the", "agent", "joins."],
        ["the", "agent", "joins."],
        ["Next", "sentence", "here."],
        ["After", "that."],
    ]
    words = _indexed(*[_words(group) for group in groups])
    first = 2
    decision = _one(_long_words()).model_copy(
        update={"dropped_words": (first, first + 2), "kept_from_word": first + 3}
    )
    _judged, client = _judge(decision, words, _verdict("both", 0.9))

    assert client.calls[0][1] == "\n".join(
        [
            "Before:",
            "Keep this.",
            "",
            "A:",
            "the agent joins.",
            "",
            "B:",
            "the agent joins. Next sentence here.",
            "",
            "After:",
            "After that.",
        ]
    )


def test_user_prompt_marks_missing_context() -> None:
    words = _long_words()
    _judged, client = _judge(_one(words), words, _verdict("both", 0.9))

    assert client.calls[0][1] == "\n".join(
        [
            "Before:",
            "(none)",
            "",
            "A:",
            "the agent joins",
            "",
            "B:",
            "the agent joins",
            "",
            "After:",
            "(none)",
        ]
    )


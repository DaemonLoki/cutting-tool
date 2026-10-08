"""Ask a local model whether an ambiguous aborted take should be cut.

Only ``long_segment`` and ``retake_missing_content`` are sent. A finished
sentence stays kept and flagged. If the endpoint is down, the decisions
already on disk are left as they are.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import JudgeConfig, Profile, load_profile
from cutter.llm import EndpointUnreachable, JsonClient, OpenAIJsonClient
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    JudgeVerdict,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)
from cutter.retakes import STAGE as RETAKES_STAGE
from cutter.retakes import STAGE_VERSION as RETAKES_STAGE_VERSION
from cutter.retakes import detect_retakes

STAGE = "judge"
STAGE_VERSION = 2

_ELIGIBLE_REASONS = frozenset({"long_segment", "retake_missing_content"})
_SENTENCE_END = (".", "?", "!")
_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "retake-judging" / "SKILL.md"
_DEFAULT_SYSTEM = (
    "You judge whether an earlier spoken attempt should be cut.\n"
    "Segment A is a candidate failed attempt. Segment B is the candidate retake.\n"
    'Answer "B" if A should be dropped. '
    'Answer "A" if B is the failed one and A should be kept instead. '
    'Answer "both" if this is not a retake.\n'
    "Reply with choice, confidence from 0 to 1, and a reason of at most 300 characters.\n"
)
_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choice", "confidence", "reason"],
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B", "both"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 300},
    },
}

logger = logging.getLogger("cutter.judge")


def judge_cached(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    no_llm: bool = False,
) -> bool:
    """True when ``decisions.json`` is already the judge result for these inputs.

    ``cutter run`` uses this to avoid replaying retakes. Replaying them would
    replace a judged artifact and force the model to run again.
    """
    loaded = load_profile("long") if profile is None else profile
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    if not words_path.is_file():
        return False
    return cache_hit(
        artifacts / "decisions.json",
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=_inputs_hash(words_path, loaded, no_llm=no_llm),
        config_hash=config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE]),
    )


def judge_decisions(
    decisions: list[Decision],
    words: list[Word],
    config: JudgeConfig,
    client: JsonClient,
) -> list[Decision]:
    """Return new Decision objects. Does not mutate the inputs."""
    judged, _unreachable = _judge_all(decisions, words, config, client)
    return judged


def run_judge(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
    no_llm: bool = False,
    client: JsonClient | None = None,
) -> DecisionsArtifact:
    """Stage entry for ``cutter judge <project_dir> [--force] [--no-llm]``.

    Reads ``artifacts/words.json``, recomputes retakes, and writes
    ``artifacts/decisions.json``. A cache hit returns the artifact already
    on disk. ``no_llm`` and a disabled judge still record a judge-stage
    artifact whose decisions are the recomputed retakes. An unreachable
    endpoint leaves the on-disk decisions artifact unchanged.
    """
    loaded = load_profile("long") if profile is None else profile
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    if not words_path.is_file():
        raise FileNotFoundError(f"missing words artifact: {words_path}")

    if not force and judge_cached(project_dir, loaded, no_llm=no_llm):
        return DecisionsArtifact.model_validate_json(decisions_path.read_text(encoding="utf-8"))

    hashed_inputs = _inputs_hash(words_path, loaded, no_llm=no_llm)
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])

    words_artifact = WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8"))
    words = words_artifact.data.words
    decisions = detect_retakes(words, loaded.retakes)
    judged, unreachable = _judged_decisions(
        decisions,
        words,
        loaded.judge,
        no_llm=no_llm,
        client=client,
    )
    if unreachable:
        return DecisionsArtifact.model_validate_json(decisions_path.read_text(encoding="utf-8"))

    artifact = DecisionsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=DecisionsData(decisions=judged),
    )
    write_artifact(decisions_path, artifact)
    return artifact


def _judged_decisions(
    decisions: list[Decision],
    words: list[Word],
    config: JudgeConfig,
    *,
    no_llm: bool,
    client: JsonClient | None,
) -> tuple[list[Decision], bool]:
    if no_llm or not config.enabled:
        return decisions, False
    if not any(_eligible(decision) for decision in decisions):
        return decisions, False
    llm = client
    if llm is None:
        llm = OpenAIJsonClient(
            base_url=config.base_url,
            model=config.model,
            timeout_s=float(config.timeout_s),
        )
    return _judge_all(decisions, words, config, llm)


def _judge_all(
    decisions: list[Decision],
    words: list[Word],
    config: JudgeConfig,
    client: JsonClient,
) -> tuple[list[Decision], bool]:
    system = _system_prompt()
    judged: list[Decision] = []
    for decision in decisions:
        if not _eligible(decision):
            judged.append(_unchanged(decision))
            continue
        try:
            payload = client.complete_json(system, _user_prompt(decision, words), _VERDICT_SCHEMA)
        except EndpointUnreachable:
            logger.warning("judge endpoint unreachable; leaving decisions unchanged")
            return [item.model_copy() for item in decisions], True
        judged.append(_with_payload(decision, payload, config))
    return judged, False


def _eligible(decision: Decision) -> bool:
    return decision.flag and decision.flag_reason in _ELIGIBLE_REASONS


def _unchanged(decision: Decision) -> Decision:
    if decision.flag_reason == "complete_sentence":
        return decision.model_copy(
            update={
                "action": "keep",
                "flag": True,
                "flag_reason": "complete_sentence",
                "judge": None,
            }
        )
    return decision.model_copy()


def _with_payload(decision: Decision, payload: object, config: JudgeConfig) -> Decision:
    verdict = _parse_verdict(payload, config.model)
    if verdict is None:
        return decision.model_copy()
    return _apply_verdict(decision, verdict, config)


def _parse_verdict(payload: object, model: str) -> JudgeVerdict | None:
    if not isinstance(payload, dict):
        return None
    choice = payload.get("choice")
    confidence = payload.get("confidence")
    reason = payload.get("reason")
    if choice not in ("A", "B", "both"):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        return None
    if not isinstance(reason, str) or len(reason) > 300:
        return None
    return JudgeVerdict(choice=choice, confidence=float(confidence), reason=reason, model=model)


def _apply_verdict(decision: Decision, verdict: JudgeVerdict, config: JudgeConfig) -> Decision:
    if verdict.confidence < config.min_confidence:
        return decision.model_copy(update={"action": "keep", "flag": True, "judge": verdict})
    if verdict.choice == "B":
        return decision.model_copy(
            update={"action": "drop", "flag": False, "flag_reason": None, "judge": verdict}
        )
    if verdict.choice == "both":
        return decision.model_copy(
            update={"action": "keep", "flag": False, "flag_reason": None, "judge": verdict}
        )
    return decision.model_copy(
        update={
            "action": "keep",
            "flag": True,
            "flag_reason": "judge_prefers_first_take",
            "judge": verdict,
        }
    )


def _system_prompt() -> str:
    if _SKILL_PATH.is_file():
        return _SKILL_PATH.read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM


def _user_prompt(decision: Decision, words: list[Word]) -> str:
    first, last = decision.dropped_words
    segment_a = words[first : last + 1]
    segment_b, b_end = _segment_b(words, decision.kept_from_word, len(segment_a))
    before = _words_before(words, first)
    after = _one_sentence(words, b_end)
    return "\n".join(
        [
            "Before:",
            _optional(before),
            "",
            "A:",
            _spoken(segment_a),
            "",
            "B:",
            _spoken(segment_b),
            "",
            "After:",
            _optional(after),
        ]
    )


def _words_before(words: list[Word], first: int) -> list[Word]:
    if first <= 0 or first >= len(words):
        return []
    sentence = words[first].sent
    earliest = sentence - 2
    return [word for word in words[:first] if earliest <= word.sent < sentence]


def _segment_b(words: list[Word], start: int, count: int) -> tuple[list[Word], int]:
    """Same-length span, then one more sentence past the current sentence end."""
    if count < 1 or start >= len(words):
        return [], min(start, len(words))
    initial_end = min(start + count, len(words))
    extra = 1 if _ends_sentence(words[initial_end - 1]) else 2
    end = _consume_sentences(words, initial_end, extra)
    return words[start:end], end


def _one_sentence(words: list[Word], start: int) -> list[Word]:
    return words[start : _consume_sentences(words, start, 1)]


def _consume_sentences(words: list[Word], start: int, needed: int) -> int:
    found = 0
    index = start
    limit = len(words)
    while index < limit and found < needed:
        if _ends_sentence(words[index]):
            found += 1
        index += 1
    return index


def _ends_sentence(word: Word) -> bool:
    return word.w.endswith(_SENTENCE_END)


def _spoken(span: list[Word]) -> str:
    return " ".join(word.w for word in span)


def _optional(span: list[Word]) -> str:
    text = _spoken(span)
    return text if text else "(none)"


def _inputs_hash(words_path: Path, profile: Profile, *, no_llm: bool) -> str:
    mode = "no_llm" if no_llm else "llm"
    base = inputs_hash(artifacts=[words_path])
    retakes_config = config_hash(profile, STAGE_CONFIG_SECTIONS[RETAKES_STAGE])
    payload = f"{base}\0{retakes_config}\0{RETAKES_STAGE_VERSION}\0{mode}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

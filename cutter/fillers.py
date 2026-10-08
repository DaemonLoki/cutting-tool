"""Mark configured filler words, phrases, and sentence openers.

Kept words are those no ``drop`` decision covers. A filler that is the only
kept word of its sentence stays; it is content. Tighten does not read this
artifact yet, so the rough cut still plays the marked words.
"""

from __future__ import annotations

from pathlib import Path

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import FillersConfig, Profile, load_profile
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)

STAGE = "fillers"
STAGE_VERSION = 2


def detect_fillers(
    words: list[Word],
    decisions: list[Decision],
    config: FillersConfig,
) -> list[Decision]:
    """Return filler decisions for kept words, in transcript order.

    A run of consecutive kept filler words is one decision. Exact ``phrases``
    are one decision and are not also split into single-word decisions. A
    ``sentence_start_words`` entry matches when it opens its sentence and the
    next kept word stays in that sentence. Longer than ``max_duration_s`` is
    kept and flagged ``filler_long``.
    """
    if not config.enabled:
        return []
    kept = [word for word in words if word.i not in _dropped_indexes(decisions)]
    if not kept:
        return []

    sole = _sole_sentence_words(kept)
    phrases = _phrases(config.phrases)
    filler_norms = set(config.words)
    openers = set(config.sentence_start_words)
    spans = _spans(kept, sole, phrases, filler_norms, openers)
    return [
        _decision(kept, start, end, number, config.max_duration_s)
        for number, (start, end) in enumerate(spans, start=1)
    ]


def run_fillers(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> DecisionsArtifact:
    """Stage entry for ``cutter fillers <project_dir> [--profile] [--force]``.

    Reads ``artifacts/words.json`` and ``artifacts/decisions.json``. Writes
    ``artifacts/fillers.json``. A cache hit returns the artifact already on
    disk. The hash covers those two artifacts and the ``fillers`` config.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    fillers_path = artifacts / "fillers.json"
    for label, path in (("words", words_path), ("decisions", decisions_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} artifact: {path}")

    hashed_inputs = inputs_hash(artifacts=[words_path, decisions_path])
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        fillers_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return DecisionsArtifact.model_validate_json(fillers_path.read_text(encoding="utf-8"))

    words = WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8"))
    decided = DecisionsArtifact.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    artifact = DecisionsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=DecisionsData(
            decisions=detect_fillers(words.data.words, decided.data.decisions, loaded.fillers)
        ),
    )
    write_artifact(fillers_path, artifact)
    return artifact


def _dropped_indexes(decisions: list[Decision]) -> set[int]:
    """Word indexes covered by an inclusive ``drop`` range.

    ``keep`` does not remove the words, so a flagged retake can still be a filler.
    """
    dropped: set[int] = set()
    for decision in decisions:
        if decision.action != "drop":
            continue
        first, last = decision.dropped_words
        dropped.update(range(first, last + 1))
    return dropped


def _sole_sentence_words(kept: list[Word]) -> set[int]:
    """Indexes that are the only kept word of their sentence.

    ``sent`` already advances at a source boundary, so one index is one sentence.
    """
    counts: dict[int, int] = {}
    for word in kept:
        counts[word.sent] = counts.get(word.sent, 0) + 1
    return {word.i for word in kept if counts[word.sent] == 1}


def _phrases(phrases: list[str]) -> tuple[tuple[str, ...], ...]:
    """Exact norm sequences, longest first so a longer phrase wins a tie."""
    compiled = [tuple(part for part in phrase.split() if part) for phrase in phrases]
    compiled = [phrase for phrase in compiled if phrase]
    compiled.sort(key=len, reverse=True)
    return tuple(compiled)


def _spans(
    kept: list[Word],
    sole: set[int],
    phrases: tuple[tuple[str, ...], ...],
    filler_norms: set[str],
    openers: set[str],
) -> list[tuple[int, int]]:
    """Inclusive indexes into ``kept``, in transcript order."""
    covered = [False] * len(kept)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(kept):
        if kept[index].i in sole:
            index += 1
            continue
        phrase_end = _phrase_end(kept, index, phrases, sole)
        if phrase_end is not None:
            _take(spans, covered, index, phrase_end)
            index = phrase_end + 1
            continue
        if kept[index].norm in filler_norms:
            end = index
            while _continues_run(kept, end, sole, filler_norms):
                end += 1
            _take(spans, covered, index, end)
            index = end + 1
            continue
        index += 1

    for index, word in enumerate(kept):
        if covered[index] or word.i in sole or word.norm not in openers:
            continue
        if not _opens_sentence(kept, index, covered):
            continue
        if index + 1 >= len(kept) or kept[index + 1].sent != word.sent:
            continue
        spans.append((index, index))
    spans.sort()
    return spans


def _phrase_end(
    kept: list[Word],
    index: int,
    phrases: tuple[tuple[str, ...], ...],
    sole: set[int],
) -> int | None:
    for phrase in phrases:
        end = index + len(phrase) - 1
        if end >= len(kept):
            continue
        window = kept[index : end + 1]
        if any(word.i in sole for word in window):
            continue
        if any(word.source != window[0].source for word in window):
            continue
        if tuple(word.norm for word in window) == phrase:
            return end
    return None


def _continues_run(
    kept: list[Word],
    end: int,
    sole: set[int],
    filler_norms: set[str],
) -> bool:
    nxt = end + 1
    if nxt >= len(kept):
        return False
    word = kept[nxt]
    return word.i not in sole and word.norm in filler_norms and word.source == kept[end].source


def _take(spans: list[tuple[int, int]], covered: list[bool], start: int, end: int) -> None:
    spans.append((start, end))
    for index in range(start, end + 1):
        covered[index] = True


def _opens_sentence(kept: list[Word], index: int, covered: list[bool]) -> bool:
    """True when every earlier kept word of this sentence is already a filler.

    ``Um, so the agent`` drops ``so`` once it is listed in
    ``sentence_start_words``, because ``um`` is a filler match. A mid-sentence
    ``so`` stays. Sentence-start marks are not filler matches, so ``so well``
    marks only ``so``.
    """
    sentence = kept[index].sent
    earlier = index - 1
    while earlier >= 0 and kept[earlier].sent == sentence:
        if not covered[earlier]:
            return False
        earlier -= 1
    return True


def _decision(
    kept: list[Word],
    start: int,
    end: int,
    number: int,
    max_duration_s: float,
) -> Decision:
    first = kept[start]
    last = kept[end]
    duration = last.end - first.start
    long = duration > max_duration_s
    return Decision(
        id=f"f{number:03d}",
        kind="filler",
        dropped_words=(first.i, last.i),
        kept_from_word=last.i + 1,
        match_words=0,
        dropped_duration_s=duration,
        action="keep" if long else "drop",
        flag=long,
        flag_reason="filler_long" if long else None,
        judge=None,
        evidence="transcript",
        clap_s=None,
        score_gap=None,
    )

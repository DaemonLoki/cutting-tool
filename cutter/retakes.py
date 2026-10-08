"""Find a retake: a later take that repeats an earlier take.

The later take stays. An aborted failed take is dropped. An exact repeat of
one finished sentence is dropped too. A finished sentence followed by
different words stays ``keep`` with reason ``complete_sentence``.
"""

from __future__ import annotations

from pathlib import Path

from rapidfuzz import fuzz

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, RetakesConfig, load_profile
from cutter.models import (
    Decision,
    DecisionsArtifact,
    DecisionsData,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)

STAGE_VERSION = 2
STAGE = "retakes"

# Common English function words. ``stopwords_file: null`` uses this list.
_BUILTIN_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "so",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "is",
        "it",
        "that",
        "this",
        "as",
        "be",
        "are",
        "was",
        "if",
        "then",
    }
)
_SENTENCE_END = (".", "?", "!")


def detect_retakes(words: list[Word], config: RetakesConfig) -> list[Decision]:
    """Return one decision per candidate repeat, in the order they are found.

    The scan walks left to right. The nearest earlier run that matches wins.
    Words already covered by a decision are not the start of another one, so
    one repeated sentence produces one decision. A dropped failed take is not
    matched again. Cross-file retakes are recorded after that scan: the start
    of source ``k+1`` against the tail of source ``k``.
    """
    stopwords = _load_stopwords(config)
    dropped: set[int] = set()
    claimed: set[int] = set()
    decisions: list[Decision] = []

    def record(j: int, i: int, match_words: int) -> None:
        decision = _make_decision(words, j, i, match_words, config, stopwords, len(decisions) + 1)
        decisions.append(decision)
        claimed.update(range(j, i))
        if decision.action == "drop":
            dropped.update(range(j, i))

    for i in range(len(words)):
        if i in dropped:
            continue
        found = _first_match(words, i, _candidates(words, i, claimed, config), config, stopwords)
        if found is not None:
            record(found[0], i, found[1])

    for i, candidates in _cross_file_candidates(words, claimed, config):
        found = _first_match(words, i, candidates, config, stopwords)
        if found is not None:
            record(found[0], i, found[1])
    return decisions


def run_retakes(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> DecisionsArtifact:
    """Stage entry for ``cutter retakes <project_dir> [--force]``.

    Reads ``artifacts/words.json`` and writes ``artifacts/decisions.json``.
    Pass ``profile`` from ``cutter run`` when it is not the default ``long``
    profile. A cache hit returns the artifact already on disk.
    """
    loaded = load_profile("long") if profile is None else profile
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    hashed_inputs = inputs_hash(artifacts=[words_path])
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        decisions_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return DecisionsArtifact.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    words_artifact = WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8"))
    artifact = DecisionsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=DecisionsData(decisions=detect_retakes(words_artifact.data.words, loaded.retakes)),
    )
    write_artifact(decisions_path, artifact)
    return artifact


def _candidates(
    words: list[Word],
    i: int,
    claimed: set[int],
    config: RetakesConfig,
):
    """Earlier indices in this source, nearest first, inside the lookback."""
    origin = words[i].start
    source = words[i].source
    for j in range(i - 1, -1, -1):
        if words[j].source != source:
            break
        if origin - words[j].start > config.max_lookback_s:
            break
        if j in claimed:
            continue
        yield j


def _cross_file_candidates(
    words: list[Word],
    claimed: set[int],
    config: RetakesConfig,
) -> list[tuple[int, list[int]]]:
    """Pair each source tail with the first word of the next source.

    Lookback is measured on source ``k`` alone: ``j`` must start within
    ``max_lookback_s`` of that source's last word. Word times are per source,
    so they are not compared across the boundary.
    """
    scans: list[tuple[int, list[int]]] = []
    spans = _source_spans(words)
    for (k_start, k_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        last_end = words[k_end].end
        candidates: list[int] = []
        for j in range(k_end, k_start - 1, -1):
            if last_end - words[j].start > config.max_lookback_s:
                break
            if j in claimed:
                continue
            candidates.append(j)
        scans.append((next_start, candidates))
    return scans


def _source_spans(words: list[Word]) -> list[tuple[int, int]]:
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words)):
        if words[index].source != words[start].source:
            spans.append((start, index - 1))
            start = index
    spans.append((start, len(words) - 1))
    return spans


def _first_match(
    words: list[Word],
    i: int,
    candidates,
    config: RetakesConfig,
    stopwords: frozenset[str],
) -> tuple[int, int] | None:
    for j in candidates:
        match_words, mismatches, matched = _match_length(words, j, i, config)
        if (
            match_words >= config.min_match_words
            and mismatches <= config.max_mismatches
            and _has_content_word(matched, match_words, stopwords)
        ):
            return j, match_words
    return None


def _match_length(
    words: list[Word],
    j: int,
    i: int,
    config: RetakesConfig,
) -> tuple[int, int, list[str]]:
    """Walk ``words[j+t]`` against ``words[i+t]`` from the start of both runs.

    A mismatch counts only when the next comparison matches, and only up to
    ``max_mismatches``. The first mismatch that is not immediately followed by
    a match ends the run. Match count does not include those skipped words.
    Each side stays in its own source, so the main scan cannot read the next
    file. A cross-file retake still compares the two sources, because ``i`` is
    the first word of the later one.
    """
    matches = 0
    mismatches = 0
    matched: list[str] = []
    t = 0
    n = len(words)
    limit = config.window_words
    threshold = config.word_similarity
    earlier = words[j].source
    later = words[i].source
    while (
        t < limit
        and j + t < i
        and i + t < n
        and words[j + t].source == earlier
        and words[i + t].source == later
    ):
        if _similar(words[j + t].norm, words[i + t].norm, threshold):
            matches += 1
            matched.append(words[j + t].norm)
            t += 1
            continue
        nxt = t + 1
        followed_by_match = (
            mismatches < config.max_mismatches
            and nxt < limit
            and j + nxt < i
            and i + nxt < n
            and words[j + nxt].source == earlier
            and words[i + nxt].source == later
            and _similar(words[j + nxt].norm, words[i + nxt].norm, threshold)
        )
        if not followed_by_match:
            break
        mismatches += 1
        t += 1
    return matches, mismatches, matched


def _similar(left: str, right: str, threshold: int) -> bool:
    return fuzz.ratio(left, right) >= threshold


def _has_content_word(matched: list[str], match_words: int, stopwords: frozenset[str]) -> bool:
    """Reject a run that is only function words, unless it is at least 5 long."""
    if match_words >= 5:
        return True
    return any(word not in stopwords for word in matched)


def _make_decision(
    words: list[Word],
    j: int,
    i: int,
    match_words: int,
    config: RetakesConfig,
    stopwords: frozenset[str],
    number: int,
) -> Decision:
    last = i - 1
    duration = words[last].end - words[j].start
    action, flag, reason = _guards(words, j, i, duration, config, stopwords)
    return Decision(
        id=f"d{number:03d}",
        kind="retake",
        dropped_words=(j, last),
        kept_from_word=i,
        match_words=match_words,
        dropped_duration_s=duration,
        action=action,
        flag=flag,
        flag_reason=reason,
        judge=None,
    )


def _guards(
    words: list[Word],
    j: int,
    i: int,
    duration: float,
    config: RetakesConfig,
    stopwords: frozenset[str],
) -> tuple[str, bool, str | None]:
    """First matching guard wins.

    An exact repeat of one finished sentence is dropped. Any other earlier
    span that contains a sentence end stays, and the judge cannot drop it.
    """
    earlier = words[j:i]
    if _exact_finished_repeat(words, j, i):
        return "drop", False, None
    if any(word.w.endswith(_SENTENCE_END) for word in earlier):
        return "keep", True, "complete_sentence"
    if duration > config.auto_drop_max_s:
        return "keep", True, "long_segment"
    # D and K are sets of normalized content words, so a repeated word counts
    # once. |D − K| / |D| is the share of the failed take's content that the
    # later take's window does not contain. An empty D skips this guard.
    dropped_content = _content_norms(earlier, stopwords)
    if dropped_content:
        dropped_len = i - j
        kept = words[i : i + dropped_len + 10]
        kept_content = _content_norms(kept, stopwords)
        missing = len(dropped_content - kept_content) / len(dropped_content)
        if missing > config.missing_content_ratio:
            return "drop", True, "retake_missing_content"
    return "drop", False, None


def _exact_finished_repeat(words: list[Word], j: int, i: int) -> bool:
    """True when ``words[j:i]`` and the sentence at ``i`` are the same words.

    Both spans are one sentence: only the last word ends in ``.``, ``?``, or
    ``!``, and the normalized tokens match. A longer or shorter later sentence
    is not an exact repeat.
    """
    earlier = words[j:i]
    if not earlier or not _ends_sentence(earlier[-1]):
        return False
    if any(_ends_sentence(word) for word in earlier[:-1]):
        return False
    end = i + len(earlier)
    if end > len(words):
        return False
    later = words[i:end]
    if any(word.source != words[i].source for word in later):
        return False
    if any(_ends_sentence(word) for word in later[:-1]) or not _ends_sentence(later[-1]):
        return False
    return [word.norm for word in earlier] == [word.norm for word in later]


def _ends_sentence(word: Word) -> bool:
    return word.w.endswith(_SENTENCE_END)


def _content_norms(span: list[Word], stopwords: frozenset[str]) -> set[str]:
    return {word.norm for word in span if word.norm and word.norm not in stopwords}


def _load_stopwords(config: RetakesConfig) -> frozenset[str]:
    if config.stopwords_file is None:
        return _BUILTIN_STOPWORDS
    words: list[str] = []
    for line in Path(config.stopwords_file).read_text(encoding="utf-8").splitlines():
        word = line.strip().casefold()
        if word:
            words.append(word)
    return frozenset(words)

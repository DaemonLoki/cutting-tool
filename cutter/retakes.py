"""Find retakes from the transcript, a script, and claps.

Without a script, the later take stays. An aborted failed take is dropped.
An exact repeat of one finished sentence is dropped too. A finished sentence
followed by different words stays ``keep`` with reason ``complete_sentence``.

With a script, the chosen take stays. A clap anchors a shorter match. A clap
with no retake after it is kept and flagged ``clap_without_retake``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from rapidfuzz import fuzz

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import ClapsConfig, Profile, RetakesConfig, ScriptConfig, load_profile
from cutter.models import (
    AlignmentArtifact,
    AlignmentData,
    AudioEventsArtifact,
    Clap,
    Decision,
    DecisionsArtifact,
    DecisionsData,
    Take,
    Word,
    WordsArtifact,
    make_meta,
    write_artifact,
)

STAGE_VERSION = 4
STAGE = "retakes"
logger = logging.getLogger("cutter.retakes")

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


def detect_retakes(
    words: list[Word],
    config: RetakesConfig,
    *,
    claps: Sequence[Clap] | None = None,
    claps_config: ClapsConfig | None = None,
    alignment: AlignmentData | None = None,
    script_config: ScriptConfig | None = None,
) -> list[Decision]:
    """Return one decision per candidate repeat, in the order they are found.

    Script decisions come first, and only when ``alignment`` has a chosen take.
    The transcript scan is the Phase 1 walk: left to right, nearest earlier run
    wins, with ``claimed`` already holding those script spans. Clap anchors
    follow, unless ``claps_config.enabled`` is false or no claps were passed.
    Cross-file retakes are last: the start of source ``k+1`` against the tail
    of source ``k``. Omit the optional arguments for the transcript scan alone.
    """
    stopwords = _load_stopwords(config)
    dropped: set[int] = set()
    claimed: set[int] = set()
    decisions: list[Decision] = []

    def record(decision: Decision) -> None:
        decisions.append(decision)
        first, last = decision.dropped_words
        claimed.update(range(first, last + 1))
        if decision.action == "drop":
            dropped.update(range(first, last + 1))

    if alignment is not None and script_config is not None:
        scripted = _script_decisions(
            words, alignment, config, script_config, claimed, decisions
        )
        for decision in scripted:
            record(decision)

    for i in range(len(words)):
        if i in dropped or i in claimed:
            continue
        found = _first_match(
            words,
            i,
            _candidates(words, i, claimed, config),
            config,
            stopwords,
            claimed,
        )
        if found is not None:
            j, match_words = found
            record(_make_decision(words, j, i, match_words, config, stopwords, len(decisions) + 1))

    if claps and claps_config is not None and claps_config.enabled:
        for decision in _clap_decisions(
            words, claps, config, claps_config, stopwords, claimed, decisions
        ):
            record(decision)

    for i, candidates in _cross_file_candidates(words, claimed, config):
        if i in dropped or i in claimed:
            continue
        found = _first_match(words, i, candidates, config, stopwords, claimed)
        if found is not None:
            j, match_words = found
            record(_make_decision(words, j, i, match_words, config, stopwords, len(decisions) + 1))
    return decisions


def run_retakes(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> DecisionsArtifact:
    """Stage entry for ``cutter retakes <project_dir> [--force]``.

    Reads ``artifacts/words.json``. Also reads ``artifacts/audio_events.json``
    and ``artifacts/alignment.json`` when those files exist. A missing file
    leaves that evidence out. Writes ``artifacts/decisions.json``. The input
    hash covers the files that exist. Pass ``profile`` from ``cutter run``
    when it is not the default ``long`` profile. A cache hit returns the
    artifact already on disk.
    """
    loaded = load_profile("long") if profile is None else profile
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    decisions_path = artifacts / "decisions.json"
    hashed_inputs = inputs_hash(artifacts=hashed_retake_paths(artifacts))
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
        data=DecisionsData(
            decisions=detect_retakes(
                words_artifact.data.words,
                loaded.retakes,
                **evidence_kwargs(artifacts, loaded),
            )
        ),
    )
    write_artifact(decisions_path, artifact)
    return artifact


def hashed_retake_paths(artifacts: Path) -> list[Path]:
    """``words.json``, plus audio events and alignment when those files exist."""
    paths = [artifacts / "words.json"]
    for name in ("audio_events.json", "alignment.json"):
        path = artifacts / name
        if path.is_file():
            paths.append(path)
    return paths


def evidence_kwargs(artifacts: Path, profile: Profile) -> dict[str, object]:
    """Claps and alignment for ``detect_retakes``. A missing file is omitted."""
    events_path = artifacts / "audio_events.json"
    alignment_path = artifacts / "alignment.json"
    events = (
        AudioEventsArtifact.model_validate_json(events_path.read_text(encoding="utf-8"))
        if events_path.is_file()
        else None
    )
    alignment = (
        AlignmentArtifact.model_validate_json(alignment_path.read_text(encoding="utf-8"))
        if alignment_path.is_file()
        else None
    )
    return {
        "claps": None if events is None else events.data.claps,
        "claps_config": profile.claps,
        "alignment": None if alignment is None else alignment.data,
        "script_config": profile.script,
    }


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
    claimed: set[int],
    *,
    min_match_words: int | None = None,
) -> tuple[int, int] | None:
    minimum = config.min_match_words if min_match_words is None else min_match_words
    for j in candidates:
        if not claimed.isdisjoint(range(j, i)):
            continue
        match_words, mismatches, matched = _match_length(words, j, i, config)
        if (
            match_words >= minimum
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


def _script_decisions(
    words: list[Word],
    alignment: AlignmentData,
    config: RetakesConfig,
    script_config: ScriptConfig,
    claimed: set[int],
    decisions: list[Decision],
):
    """Drop each alternate take. The chosen take stays, and Phase 1 guards do not apply."""
    by_i = {word.i: index for index, word in enumerate(words)}
    resolved = _resolved_takes(words, alignment, by_i)
    items: list[tuple[tuple[int, int], int, float]] = []
    for sentence in alignment.sentences:
        chosen = _chosen_take(sentence.takes)
        if chosen is None:
            continue
        chosen_span = _resolve_take(words, by_i, chosen)
        if chosen_span is None:
            continue
        for take in sentence.takes:
            if take.chosen:
                continue
            alternate = _resolve_take(words, by_i, take)
            if alternate is None or _overlaps(alternate, chosen_span):
                continue
            if _overlaps_any(alternate, resolved, skip=alternate):
                continue
            items.append((alternate, chosen_span[0], chosen.score - take.score))
    items.sort(key=lambda item: item[0][0])
    for alternate, kept_from, gap in items:
        span = _alternate_span(words, alternate, resolved, config.window_words, claimed)
        if span is None or kept_from in range(span[0], span[1] + 1):
            continue
        first, last = span
        action, flag, reason = _script_verdict(alternate[0], kept_from, gap, script_config)
        yield Decision(
            id=f"d{len(decisions) + 1:03d}",
            kind="retake",
            dropped_words=(first, last),
            kept_from_word=kept_from,
            match_words=alternate[1] - alternate[0] + 1,
            dropped_duration_s=words[last].end - words[first].start,
            action=action,
            flag=flag,
            flag_reason=reason,
            judge=None,
            evidence="script",
            clap_s=None,
            score_gap=gap,
        )


def _chosen_take(takes: Sequence[Take]) -> Take | None:
    chosen = [take for take in takes if take.chosen]
    if not chosen:
        return None
    return max(chosen, key=lambda take: (take.first_word, take.score))


def _resolved_takes(
    words: list[Word],
    alignment: AlignmentData,
    by_i: dict[int, int],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for sentence in alignment.sentences:
        for take in sentence.takes:
            span = _resolve_take(words, by_i, take)
            if span is not None:
                spans.append(span)
    spans.sort()
    return spans


def _resolve_take(
    words: list[Word],
    by_i: dict[int, int],
    take: Take,
) -> tuple[int, int] | None:
    first = by_i.get(take.first_word)
    last = by_i.get(take.last_word)
    if first is None or last is None or first > last:
        return None
    source = words[first].source
    if any(word.source != source for word in words[first : last + 1]):
        return None
    return first, last


def _alternate_span(
    words: list[Word],
    alternate: tuple[int, int],
    takes: Sequence[tuple[int, int]],
    window_words: int,
    claimed: set[int],
) -> tuple[int, int] | None:
    """Inclusive span of an alternate, plus a short broken-off tail before the next take."""
    first, last = alternate
    if any(index in claimed for index in range(first, last + 1)):
        return None
    source = words[first].source
    next_first = next(
        (start for start, _end in takes if start > last and words[start].source == source),
        None,
    )
    end = last
    if next_first is not None:
        extra: list[int] = []
        for index in range(last + 1, next_first):
            if words[index].source != source or index in claimed or _in_take(index, takes):
                break
            extra.append(index)
        if len(extra) < window_words:
            end = last + len(extra)
    return first, end


def _script_verdict(
    alternate_first: int,
    chosen_first: int,
    gap: float,
    script_config: ScriptConfig,
) -> tuple[str, bool, str | None]:
    later = alternate_first > chosen_first
    if later and not script_config.drop_later_takes:
        return "keep", True, "script_close_call"
    if gap < script_config.min_score_gap:
        return "drop", True, "script_close_call"
    return "drop", False, None


def _in_take(index: int, takes: Sequence[tuple[int, int]]) -> bool:
    return any(first <= index <= last for first, last in takes)


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _overlaps_any(
    span: tuple[int, int],
    takes: Sequence[tuple[int, int]],
    *,
    skip: tuple[int, int],
) -> bool:
    return any(take != skip and _overlaps(span, take) for take in takes)


def _clap_decisions(
    words: list[Word],
    claps: Sequence[Clap],
    config: RetakesConfig,
    claps_config: ClapsConfig,
    stopwords: frozenset[str],
    claimed: set[int],
    decisions: list[Decision],
):
    """Anchor a retake on each clap, or flag the words since the previous sentence."""
    order = _source_order(words)
    previous: dict[str, int] = {}
    for clap in sorted(claps, key=lambda item: (order.get(item.source, len(order)), item.t)):
        i = _first_word_at_or_after(words, clap.source, clap.t)
        if i is None:
            logger.info("clap at %.3fs in source %s has no following word", clap.t, clap.source)
            continue
        anchor = previous.get(clap.source)
        previous[clap.source] = i
        if i in claimed or any(decision.kept_from_word == i for decision in decisions):
            continue
        found = _first_match(
            words,
            i,
            _candidates(words, i, claimed, config),
            config,
            stopwords,
            claimed,
            min_match_words=claps_config.min_match_words,
        )
        if found is not None:
            j, match_words = found
            decision = _make_decision(
                words, j, i, match_words, config, stopwords, len(decisions) + 1
            )
            yield decision.model_copy(update={"evidence": "clap", "clap_s": clap.t})
            continue
        k = _past_claimed(_clap_span_start(words, clap.source, i, anchor), i, claimed)
        if k > i - 1:
            logger.info(
                "clap at %.3fs in source %s has no words before it to review",
                clap.t,
                clap.source,
            )
            continue
        last = i - 1
        yield Decision(
            id=f"d{len(decisions) + 1:03d}",
            kind="retake",
            dropped_words=(k, last),
            kept_from_word=i,
            match_words=0,
            dropped_duration_s=words[last].end - words[k].start,
            action="keep",
            flag=True,
            flag_reason="clap_without_retake",
            judge=None,
            evidence="clap",
            clap_s=clap.t,
            score_gap=None,
        )


def _source_order(words: Sequence[Word]) -> dict[str, int]:
    order: dict[str, int] = {}
    for word in words:
        if word.source not in order:
            order[word.source] = len(order)
    return order


def _source_bounds(words: Sequence[Word], source: str) -> tuple[int, int] | None:
    for start, end in _source_spans(list(words)):
        if words[start].source == source:
            return start, end
    return None


def _first_word_at_or_after(words: Sequence[Word], source: str, t: float) -> int | None:
    bounds = _source_bounds(words, source)
    if bounds is None:
        return None
    start, end = bounds
    for index in range(start, end + 1):
        if words[index].start >= t:
            return index
    return None


def _clap_span_start(
    words: Sequence[Word],
    source: str,
    i: int,
    previous_clap: int | None,
) -> int:
    """First word after the previous sentence end, the previous clap, or the source start."""
    bounds = _source_bounds(words, source)
    if bounds is None:
        return i
    start, _end = bounds
    k = start
    for index in range(start, i):
        if _ends_sentence(words[index]):
            k = index + 1
    if previous_clap is not None:
        k = max(k, previous_clap)
    return k


def _past_claimed(start: int, end: int, claimed: set[int]) -> int:
    """Move ``start`` past claimed words so ``[start, end)`` does not overlap a decision."""
    blocked = [index for index in range(start, end) if index in claimed]
    if not blocked:
        return start
    return blocked[-1] + 1

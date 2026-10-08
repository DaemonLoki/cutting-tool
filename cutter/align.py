"""Align transcript words to an optional script.

Reads ``artifacts/words.json`` and ``<project>/<script.path>`` when that file
exists. Writes ``artifacts/alignment.json``. A missing script stores
``script: null`` and empty lists.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, load_profile
from cutter.models import (
    AlignmentArtifact,
    AlignmentData,
    ScriptChapter,
    ScriptInfo,
    ScriptSentence,
    Take,
    Word,
    WordsArtifact,
    WordSpan,
    make_meta,
    write_artifact,
)
from cutter.script import parse_script, script_bytes_hash

STAGE = "align"
STAGE_VERSION = 2

logger = logging.getLogger("cutter.align")


@dataclass(frozen=True)
class _Window:
    first_word: int
    last_word: int
    score: float


@dataclass(frozen=True)
class _Chain:
    objective: float
    flags: tuple[tuple[int, int], ...]
    parent: tuple[int, int] | None


def align_script(
    words: Sequence[Word],
    markdown: str,
    *,
    levels: Sequence[int],
    word_similarity: int,
    min_take_score: int,
    prefer: Literal["best", "last"],
    path: str,
    script_hash: str,
) -> AlignmentData:
    """Match ``markdown`` to ``words`` and choose one monotonic take per sentence.

    ``word_similarity`` is the rapidfuzz ratio a transcript word needs against
    the sentence's first word, or its second word when the sentence is longer.
    ``min_take_score`` is the minimum ratio for a whole window. ``prefer``
    ``best`` maximizes the sum of those scores. ``last`` maximizes the sum of
    ``first_word`` instead. Ties keep the later take.
    """
    parsed = parse_script(markdown, levels=levels)
    ordered = sorted(words, key=lambda word: word.i)
    windows = [
        _candidate_windows(
            ordered,
            sentence.words,
            word_similarity=word_similarity,
            min_take_score=min_take_score,
        )
        for sentence in parsed.sentences
    ]
    chosen = _choose_takes(windows, prefer)
    sentences: list[ScriptSentence] = []
    kept: list[_Window] = []
    for index, sentence in enumerate(parsed.sentences):
        pick = chosen.get(index)
        if pick is None:
            takes: list[Take] = []
        else:
            takes = [
                Take(
                    first_word=window.first_word,
                    last_word=window.last_word,
                    score=window.score,
                    chosen=offset == pick,
                )
                for offset, window in enumerate(windows[index])
            ]
            kept.extend(windows[index])
        sentences.append(
            ScriptSentence(
                id=sentence.id,
                chapter=sentence.chapter,
                text=sentence.text,
                takes=takes,
            )
        )
    return AlignmentData(
        script=ScriptInfo(path=path, hash=script_hash),
        chapters=[
            ScriptChapter(
                id=chapter.id,
                title=chapter.title,
                level=chapter.level,
                first_sentence=chapter.first_sentence,
            )
            for chapter in parsed.chapters
        ],
        sentences=sentences,
        unscripted=_unscripted(ordered, kept),
        missing=[sentence.id for sentence in parsed.sentences if sentence.id not in chosen],
    )


def run_align(
    project_dir: Path,
    profile: Profile | None = None,
    *,
    force: bool = False,
) -> AlignmentArtifact:
    """Stage entry for ``cutter align <project_dir> [--profile] [--force]``.

    Reads ``artifacts/words.json`` and ``<project>/<script.path>`` when that
    file exists. Writes ``artifacts/alignment.json``. A cache hit returns the
    artifact already on disk. A missing script is logged once.
    """
    loaded = load_profile("long") if profile is None else profile
    project_dir = Path(project_dir)
    artifacts = project_dir / "artifacts"
    words_path = artifacts / "words.json"
    alignment_path = artifacts / "alignment.json"
    if not words_path.is_file():
        raise FileNotFoundError(f"missing words artifact: {words_path}")

    script_path = project_dir / loaded.script.path
    hashed = [words_path]
    if script_path.is_file():
        hashed.append(script_path)
    hashed_inputs = inputs_hash(artifacts=hashed)
    hashed_config = config_hash(loaded, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        alignment_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return AlignmentArtifact.model_validate_json(alignment_path.read_text(encoding="utf-8"))

    if not script_path.is_file():
        logger.info("no script at %s", script_path)
        data = AlignmentData(
            script=None,
            chapters=[],
            sentences=[],
            unscripted=[],
            missing=[],
        )
    else:
        raw = script_path.read_bytes()
        words = WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8"))
        data = align_script(
            words.data.words,
            raw.decode("utf-8"),
            levels=loaded.chapters.levels,
            word_similarity=loaded.retakes.word_similarity,
            min_take_score=loaded.script.min_take_score,
            prefer=loaded.script.prefer,
            path=loaded.script.path,
            script_hash=script_bytes_hash(raw),
        )
    artifact = AlignmentArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=data,
    )
    write_artifact(alignment_path, artifact)
    return artifact


def _candidate_windows(
    words: Sequence[Word],
    script_words: Sequence[str],
    *,
    word_similarity: int,
    min_take_score: int,
) -> list[_Window]:
    """Windows for one sentence, with overlaps reduced to the better take."""
    count = len(script_words)
    if count == 0 or not words:
        return []
    slack = max(1, count // 4)
    target = " ".join(script_words)
    found: list[_Window] = []
    for start, word in enumerate(words):
        if not _opens(word.norm, script_words, word_similarity):
            continue
        end_limit = _source_end(words, start, count + slack)
        best_end = start
        best_rank: tuple[float, int, int] | None = None
        for end in range(end_limit, start, -1):
            length = end - start
            joined = " ".join(words[index].norm for index in range(start, end))
            score = float(fuzz.ratio(target, joined))
            rank = (score, -abs(length - count), length)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_end = end
        if best_rank is not None and best_rank[0] >= min_take_score:
            found.append(
                _Window(
                    first_word=words[start].i,
                    last_word=words[best_end - 1].i,
                    score=best_rank[0],
                )
            )
    return _suppress_overlaps(found)


def _opens(norm: str, script_words: Sequence[str], word_similarity: int) -> bool:
    if fuzz.ratio(norm, script_words[0]) >= word_similarity:
        return True
    return len(script_words) > 1 and fuzz.ratio(norm, script_words[1]) >= word_similarity


def _source_end(words: Sequence[Word], start: int, length: int) -> int:
    """Exclusive end of a window that stays inside ``words[start]``'s source."""
    source = words[start].source
    end = start + 1
    limit = min(len(words), start + length)
    while end < limit and words[end].source == source:
        end += 1
    return end


def _suppress_overlaps(windows: Sequence[_Window]) -> list[_Window]:
    """Keep the higher score when two windows overlap. Ties keep the later one."""
    ranked = sorted(windows, key=lambda window: (window.score, window.first_word), reverse=True)
    kept: list[_Window] = []
    for window in ranked:
        if any(_overlaps(window, other) for other in kept):
            continue
        kept.append(window)
    kept.sort(key=lambda window: window.first_word)
    return kept


def _overlaps(left: _Window, right: _Window) -> bool:
    return left.first_word <= right.last_word and right.first_word <= left.last_word


def _choose_takes(
    candidates: Sequence[Sequence[_Window]],
    prefer: Literal["best", "last"],
) -> dict[int, int]:
    """Pick one window per sentence so chosen takes do not overlap.

    The objective is the sum of scores, or the sum of ``first_word`` when
    ``prefer`` is ``last``. An equal objective keeps the chain that covers a
    sentence the other skips, then the chain with the later ``first_word``.
    """
    count = len(candidates)
    skip = tuple((0, -1) for _ in range(count))
    chains: list[list[_Chain]] = []
    for index, windows in enumerate(candidates):
        row: list[_Chain] = []
        for window in windows:
            gain = float(window.first_word if prefer == "last" else window.score)
            best_key = (0.0, skip)
            best_parent: tuple[int, int] | None = None
            best_flags = skip
            for earlier, earlier_row in enumerate(chains):
                for offset, earlier_window in enumerate(candidates[earlier]):
                    if window.first_word <= earlier_window.last_word:
                        continue
                    if window.first_word <= earlier_window.first_word:
                        continue
                    chain = earlier_row[offset]
                    key = (chain.objective, chain.flags)
                    if key > best_key:
                        best_key = key
                        best_parent = (earlier, offset)
                        best_flags = chain.flags
            flags = best_flags[:index] + ((1, window.first_word),) + best_flags[index + 1 :]
            row.append(_Chain(best_key[0] + gain, flags, best_parent))
        chains.append(row)

    best_key = (0.0, skip)
    best_at: tuple[int, int] | None = None
    for index, row in enumerate(chains):
        for offset, chain in enumerate(row):
            key = (chain.objective, chain.flags)
            if key > best_key:
                best_key = key
                best_at = (index, offset)
    chosen: dict[int, int] = {}
    while best_at is not None:
        index, offset = best_at
        chosen[index] = offset
        best_at = chains[index][offset].parent
    return chosen


def _unscripted(words: Sequence[Word], takes: Sequence[_Window]) -> list[WordSpan]:
    """Maximal runs of words that sit in no take, including alternates."""
    covered: set[int] = set()
    for take in takes:
        for word in words:
            if take.first_word <= word.i <= take.last_word:
                covered.add(word.i)
    spans: list[WordSpan] = []
    start: Word | None = None
    previous: Word | None = None
    for word in words:
        if word.i in covered:
            if start is not None and previous is not None:
                spans.append(WordSpan(first_word=start.i, last_word=previous.i))
            start = None
            previous = None
            continue
        if start is None:
            start = word
        previous = word
    if start is not None and previous is not None:
        spans.append(WordSpan(first_word=start.i, last_word=previous.i))
    return spans

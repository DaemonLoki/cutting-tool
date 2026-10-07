"""Transcribe each source to words with times from the start of that source.

parakeet-mlx 0.4 (installed) exposes ``from_pretrained(model_id)`` and
``BaseParakeet.transcribe(path, *, chunk_duration=None, overlap_duration=15.0)``.
The result is an ``AlignedResult`` whose ``tokens`` are ``AlignedToken``
objects: ``text``, ``start``, ``duration``, ``end`` (``start + duration``), and
``confidence``. Times are seconds from the start of the audio passed in.

``tokenizer.decode`` has already turned the SentencePiece marker ``▁`` into a
leading space, so a token like ``" the"`` starts a word and ``"ent"`` continues
one. Punctuation is its own token with no leading space (``","``, ``"."``).

The library chunks when ``chunk_duration`` is set: it slices the wav, shifts
each token back onto the full file, and merges the overlap. This module passes
the profile's ``chunk_duration_s`` and ``overlap_duration_s`` through and does
not slice the wav again.
"""

from __future__ import annotations

import logging
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile, load_profile
from cutter.models import (
    Source,
    SourcesArtifact,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)

STAGE = "transcribe"
STAGE_VERSION = 1

# A gap this large is a bad alignment. Smaller gaps are clamped and logged.
# The slack absorbs binary rounding so a gap of 50 ms still raises (0.2 - 0.15
# is just under 0.05 in IEEE floats).
_TOLERANCE_S = 0.05
_TOLERANCE_SLACK_S = 1e-9
_SENTENCE_ENDINGS = (".", "?", "!")

logger = logging.getLogger(__name__)


class TimestampError(ValueError):
    """A word time disagrees with its source by 50 ms or more."""


@dataclass(frozen=True)
class RawWord:
    """One word as a transcriber emits it, before source indexes are assigned."""

    text: str
    start: float
    end: float
    conf: float | None = None


@dataclass(frozen=True)
class Subword:
    """One model token. A leading space or ``▁`` starts a new word."""

    text: str
    start: float
    end: float
    conf: float | None = None


class Transcriber(Protocol):
    """Turns one 16 kHz wav into words. Times are seconds from the start of that wav."""

    def transcribe(self, wav_path: Path) -> list[RawWord]:
        """Transcribe ``wav_path``."""


class ParakeetTranscriber:
    """Parakeet TDT transcriber. The model loads on the first wav, not at import."""

    def __init__(self, profile: Profile) -> None:
        self._model_name = profile.transcribe.model
        self._chunk_duration_s = float(profile.transcribe.chunk_duration_s)
        self._overlap_duration_s = float(profile.transcribe.overlap_duration_s)
        self._model: object | None = None

    def transcribe(self, wav_path: Path) -> list[RawWord]:
        model = self._load()
        result = model.transcribe(
            Path(wav_path),
            chunk_duration=self._chunk_duration_s,
            overlap_duration=self._overlap_duration_s,
        )
        tokens = [
            Subword(
                text=token.text,
                start=float(token.start),
                end=float(token.end),
                conf=None if token.confidence is None else float(token.confidence),
            )
            for token in result.tokens
        ]
        return merge_subwords(tokens)

    def _load(self) -> object:
        if self._model is None:
            from parakeet_mlx import from_pretrained

            self._model = from_pretrained(self._model_name)
        return self._model


def merge_subwords(tokens: Sequence[Subword]) -> list[RawWord]:
    """Merge subword tokens into words.

    A token whose text starts with a space or a SentencePiece ``▁`` starts a
    word. The word's start is that token's start; its end is the last token's
    end. Punctuation stays on the word. Confidence is the geometric mean of
    the token confidences, or ``None`` when any token has none.
    """
    words: list[RawWord] = []
    parts: list[str] = []
    confs: list[float | None] = []
    start = 0.0
    end = 0.0

    def flush() -> None:
        nonlocal parts, confs
        text = "".join(parts)
        if text:
            words.append(RawWord(text, start, end, _confidence(confs)))
        parts = []
        confs = []

    for token in tokens:
        if parts and _starts_word(token.text):
            flush()
        if not parts:
            start = token.start
        parts.append(_piece(token.text))
        end = token.end
        confs.append(token.conf)
    flush()
    return words


def assemble_words(
    transcripts: Sequence[tuple[str, float, Sequence[RawWord]]],
) -> WordsData:
    """Build ``WordsData`` from per-source words.

    Each item is ``(source id, duration_s, words)`` in source order. ``i`` is
    a global index. ``sent`` advances after a word ending in ``.``, ``?``, or
    ``!``, and at each source boundary. Those two reasons together still
    advance the index by one.
    """
    words: list[Word] = []
    index = 0
    sent = 0
    start_new_sentence = False
    for source_id, duration_s, raw_words in transcripts:
        checked = _check_times(source_id, duration_s, raw_words)
        if words:
            start_new_sentence = True
        for raw in checked:
            if start_new_sentence:
                sent += 1
                start_new_sentence = False
            words.append(
                Word(
                    i=index,
                    source=source_id,
                    w=raw.text,
                    norm=normalize_word(raw.text),
                    start=raw.start,
                    end=raw.end,
                    conf=raw.conf,
                    sent=sent,
                )
            )
            index += 1
            if _ends_sentence(raw.text):
                start_new_sentence = True
    return WordsData(words=words)


def normalize_word(text: str) -> str:
    """Lowercase, strip punctuation, apply Unicode NFKC, and keep digits."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in normalized if not unicodedata.category(ch).startswith("P"))


def words_between(words: list[Word], from_s: float, to_s: float) -> str:
    """Lines a person can check by ear: word text, start, and end.

    Includes words whose start is in ``[from_s, to_s)``. Times are seconds
    from the start of each word's source, so the same window matches every
    source. One word per line, three decimal places, no trailing newline.
    """
    lines = [
        f"{word.w} {word.start:.3f} {word.end:.3f}" for word in words if from_s <= word.start < to_s
    ]
    return "\n".join(lines)


def transcribe_project(
    project_dir: Path,
    *,
    profile: Profile | None = None,
    force: bool = False,
    transcriber: Transcriber | None = None,
) -> WordsArtifact:
    """Read ``artifacts/sources.json``, transcribe each 16 kHz wav, write ``artifacts/words.json``.

    ``profile`` defaults to the ``long`` profile. A matching cache entry is
    returned as-is unless ``force`` is set. Pass ``transcriber`` to avoid
    loading Parakeet.
    """
    profile = load_profile() if profile is None else profile
    project_dir = Path(project_dir)
    sources_path = project_dir / "artifacts" / "sources.json"
    words_path = project_dir / "artifacts" / "words.json"
    if not sources_path.is_file():
        raise FileNotFoundError(f"missing sources artifact: {sources_path}")

    sources = SourcesArtifact.model_validate_json(sources_path.read_text(encoding="utf-8"))
    wavs = [_asr_wav(project_dir, source) for source in sources.data.sources]
    for source, wav in zip(sources.data.sources, wavs, strict=True):
        if not wav.is_file():
            raise FileNotFoundError(f"missing asr wav for {source.id}: {wav}")

    hashed_inputs = inputs_hash(artifacts=[sources_path, *wavs])
    hashed_config = config_hash(profile, STAGE_CONFIG_SECTIONS[STAGE])
    if not force and cache_hit(
        words_path,
        stage=STAGE,
        stage_version=STAGE_VERSION,
        inputs_hash=hashed_inputs,
        config_hash=hashed_config,
    ):
        return WordsArtifact.model_validate_json(words_path.read_text(encoding="utf-8"))

    engine = transcriber if transcriber is not None else ParakeetTranscriber(profile)
    transcripts = [
        (source.id, source.duration_s, engine.transcribe(wav))
        for source, wav in zip(sources.data.sources, wavs, strict=True)
    ]
    artifact = WordsArtifact(
        meta=make_meta(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            inputs_hash=hashed_inputs,
            config_hash=hashed_config,
        ),
        data=assemble_words(transcripts),
    )
    write_artifact(words_path, artifact)
    return artifact


def _starts_word(text: str) -> bool:
    return text.startswith((" ", "▁"))


def _piece(text: str) -> str:
    if _starts_word(text):
        return text[1:]
    return text


def _confidence(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    total = sum(math.log(max(value, 1e-10)) for value in values)
    return math.exp(total / len(values))


def _ends_sentence(text: str) -> bool:
    stripped = unicodedata.normalize("NFKC", text).rstrip()
    return bool(stripped) and stripped.endswith(_SENTENCE_ENDINGS)


def _check_times(source_id: str, duration_s: float, raw_words: Sequence[RawWord]) -> list[RawWord]:
    if duration_s < 0:
        raise TimestampError(f"{source_id}: duration_s is negative ({duration_s})")
    checked: list[RawWord] = []
    previous: float | None = None
    for raw in raw_words:
        start = _clamp(raw.start, 0.0, source_id=source_id, text=raw.text, kind="start", low=True)
        start = _clamp(
            start, duration_s, source_id=source_id, text=raw.text, kind="start", low=False
        )
        if previous is not None:
            start = _clamp(
                start, previous, source_id=source_id, text=raw.text, kind="start", low=True
            )
        end = _clamp(raw.end, start, source_id=source_id, text=raw.text, kind="end", low=True)
        end = _clamp(end, duration_s, source_id=source_id, text=raw.text, kind="end", low=False)
        checked.append(RawWord(raw.text, start, end, raw.conf))
        previous = start
    return checked


def _clamp(
    value: float,
    limit: float,
    *,
    source_id: str,
    text: str,
    kind: str,
    low: bool,
) -> float:
    if low:
        if value >= limit:
            return value
        gap = limit - value
    else:
        if value <= limit:
            return value
        gap = value - limit
    if gap >= _TOLERANCE_S - _TOLERANCE_SLACK_S:
        raise TimestampError(
            f"{source_id}: {kind} of {text!r} is {gap * 1000:.0f} ms out of range "
            f"({value:.3f} vs {limit:.3f})"
        )
    logger.warning(
        "%s: clamped %s of %r from %.3f to %.3f (%.0f ms)",
        source_id,
        kind,
        text,
        value,
        limit,
        gap * 1000,
    )
    return limit


def _asr_wav(project_dir: Path, source: Source) -> Path:
    path = Path(source.asr_wav)
    if path.is_absolute():
        return path
    return project_dir / path

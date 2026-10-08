"""Parse ``script.md`` into chapters, sentences, and normalized words."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from cutter.transcribe import normalize_word

_FENCE_OPEN = re.compile(r"^ {0,3}(```+|~~~+)(.*)$")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_SENTENCE_END = re.compile(r"[.?!](?=\s|$)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CODE_SPAN = re.compile(r"`+([^`]*)`+")
_TRAILING_HASHES = re.compile(r"\s+#+\s*$")
_WHITESPACE = re.compile(r"\s+")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class ParsedSentence:
    """One script sentence after markdown is removed and words are normalized."""

    id: int
    chapter: str | None
    text: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class ParsedChapter:
    """A heading whose level is configured to become a chapter."""

    id: str
    title: str
    level: int
    first_sentence: int


@dataclass(frozen=True)
class ParsedScript:
    """Chapters and sentences in document order."""

    chapters: tuple[ParsedChapter, ...]
    sentences: tuple[ParsedSentence, ...]


def script_bytes_hash(data: bytes) -> str:
    """``sha256:`` plus the hex digest of the script file bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_script(markdown: str, *, levels: Sequence[int]) -> ParsedScript:
    """Turn markdown into chapters and sentences.

    YAML front matter, fenced code, HTML comments, and blank lines are
    ignored. A heading becomes a chapter only when its level is in
    ``levels``; any other heading line is dropped. Text before the first
    chapter heading has ``chapter`` None. Sentences split at ``.``, ``?``,
    or ``!`` followed by whitespace or the end of the paragraph.
    """
    text = _strip_fences(_strip_comments(_strip_front_matter(markdown)))
    chapters: list[ParsedChapter] = []
    sentences: list[ParsedSentence] = []
    paragraph: list[str] = []
    current_chapter: str | None = None

    def flush() -> None:
        if not paragraph:
            return
        block = " ".join(part.strip() for part in paragraph if part.strip())
        paragraph.clear()
        for raw in _split_sentences(block):
            text_value, words = _sentence_words(raw)
            if not words:
                continue
            sentences.append(
                ParsedSentence(
                    id=len(sentences),
                    chapter=current_chapter,
                    text=text_value,
                    words=words,
                )
            )

    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            if level not in levels:
                continue
            chapter_id = f"c{len(chapters) + 1:02d}"
            chapters.append(
                ParsedChapter(
                    id=chapter_id,
                    title=_heading_title(heading.group(2)),
                    level=level,
                    first_sentence=len(sentences),
                )
            )
            current_chapter = chapter_id
            continue
        paragraph.append(line)
    flush()
    return ParsedScript(chapters=tuple(chapters), sentences=tuple(sentences))


def _strip_front_matter(text: str) -> str:
    """Drop a leading ``---`` block closed by ``---`` or ``...``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return "\n".join(lines[index + 1 :])
    return text


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def _strip_fences(text: str) -> str:
    """Drop fenced code blocks, including the fence lines."""
    kept: list[str] = []
    closing: re.Pattern[str] | None = None
    for line in text.splitlines():
        if closing is None:
            match = _FENCE_OPEN.match(line)
            if match:
                marker = match.group(1)
                closing = re.compile(rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$")
                continue
            kept.append(line)
            continue
        if closing.match(line):
            closing = None
    return "\n".join(kept)


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        sentence = paragraph[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _sentence_words(sentence: str) -> tuple[str, tuple[str, ...]]:
    stripped = _strip_inline(sentence)
    text = _WHITESPACE.sub(" ", stripped).strip()
    words = tuple(norm for token in text.split() if (norm := normalize_word(token)))
    return text, words


def _heading_title(raw: str) -> str:
    title = _TRAILING_HASHES.sub("", raw.strip())
    return _WHITESPACE.sub(" ", _strip_inline(title)).strip()


def _strip_inline(text: str) -> str:
    """Remove inline markdown markers and keep link and code text."""
    without_links = _LINK.sub(r"\1", _IMAGE.sub(r"\1", text))
    without_code = _CODE_SPAN.sub(r"\1", without_links)
    return without_code.replace("*", "").replace("_", "").replace("`", "")

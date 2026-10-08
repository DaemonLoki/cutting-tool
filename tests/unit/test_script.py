"""Markdown script parsing: chapters, sentences, and the file hash."""

import hashlib

from cutter.script import parse_script, script_bytes_hash


def test_front_matter_code_comments_and_blanks_are_ignored():
    text = """---
title: Talk
note: Hello from the header.
---

<!-- Secret sentence that must not appear. -->

```python
print("Ignore this sentence.")
```

# Intro

Real sentence.

"""
    parsed = parse_script(text, levels=[1, 2])
    assert len(parsed.sentences) == 1
    assert parsed.sentences[0].words == ("real", "sentence")
    assert parsed.sentences[0].chapter == "c01"
    assert parsed.chapters[0].title == "Intro"


def test_other_heading_levels_are_not_sentences():
    parsed = parse_script("### Notes about the cut\n\nHello world.\n", levels=[1, 2])
    assert parsed.chapters == ()
    assert len(parsed.sentences) == 1
    assert parsed.sentences[0].words == ("hello", "world")
    assert parsed.sentences[0].chapter is None


def test_ignored_heading_does_not_change_the_chapter():
    text = "# One\n\nAlpha.\n\n### Skip me\n\nBeta.\n"
    parsed = parse_script(text, levels=[1, 2])
    assert len(parsed.chapters) == 1
    assert [sentence.chapter for sentence in parsed.sentences] == ["c01", "c01"]
    assert parsed.sentences[1].words == ("beta",)


def test_text_before_the_first_heading_has_no_chapter():
    parsed = parse_script("Before the title.\n\n# Intro\n\nAfter it.\n", levels=[1, 2])
    assert [sentence.chapter for sentence in parsed.sentences] == [None, "c01"]
    assert parsed.chapters[0].first_sentence == 1
    assert parsed.chapters[0].id == "c01"
    assert parsed.chapters[0].title == "Intro"
    assert parsed.chapters[0].level == 1


def test_chapter_ids_and_first_sentence_follow_heading_order():
    text = "# One\n\nAlpha.\n\n## Two\n\nBeta. Gamma.\n\n# Three\n\nDelta.\n"
    parsed = parse_script(text, levels=[1, 2])
    assert [(chapter.id, chapter.level, chapter.first_sentence) for chapter in parsed.chapters] == [
        ("c01", 1, 0),
        ("c02", 2, 1),
        ("c03", 1, 3),
    ]
    assert [sentence.chapter for sentence in parsed.sentences] == ["c01", "c02", "c02", "c03"]
    assert parsed.sentences[1].words == ("beta",)
    assert parsed.sentences[2].words == ("gamma",)


def test_paragraphs_split_on_sentence_punctuation():
    parsed = parse_script("One two. Three four? Five six!\n", levels=[1])
    assert [sentence.words for sentence in parsed.sentences] == [
        ("one", "two"),
        ("three", "four"),
        ("five", "six"),
    ]
    assert [sentence.text for sentence in parsed.sentences] == [
        "One two.",
        "Three four?",
        "Five six!",
    ]


def test_inline_markdown_is_stripped_before_words():
    text = "See the [docs](https://example.com/a) and **ship** `it`.\n"
    parsed = parse_script(text, levels=[1])
    assert parsed.sentences[0].words == ("see", "the", "docs", "and", "ship", "it")
    assert parsed.sentences[0].text == "See the docs and ship it."


def test_empty_sentences_are_dropped():
    parsed = parse_script("...\n\n!!!\n\nHello.\n", levels=[1])
    assert len(parsed.sentences) == 1
    assert parsed.sentences[0].id == 0
    assert parsed.sentences[0].words == ("hello",)


def test_script_hash_is_sha256_of_the_bytes():
    raw = b"Hello.\n"
    assert script_bytes_hash(raw) == "sha256:" + hashlib.sha256(raw).hexdigest()

"""Pure parts of the synthetic project builder."""

from tests.e2e.make_fixture import (
    EXPECTED,
    FILE_ONE,
    clap_times,
    file_one_lines,
    script_markdown,
)


def test_default_lines_match_the_phase1_script():
    assert file_one_lines() == FILE_ONE
    assert "um" not in EXPECTED.casefold()
    assert "uh" not in EXPECTED.casefold()


def test_filler_line_keeps_the_same_sentence_around_the_fillers():
    lines = file_one_lines(fillers=True)
    assert lines[0] == FILE_ONE[0]
    assert lines[1] == "Um, the cache stores the result and, uh, skips the work."
    assert lines[2] == FILE_ONE[2]
    assert "um" not in EXPECTED.casefold()


def test_script_covers_both_expected_sentences_and_one_unspoken_line():
    text = script_markdown()
    assert text.startswith("# Cache\n")
    assert "# Agent\n" in text
    assert "The cache stores the result and skips the work." in text
    assert "The agent joins the call and subscribes to the audio track." in text
    assert "The agent leaves the call when the host ends it." in text
    cache, agent = text.split("# Agent\n", maxsplit=1)
    assert "The agent joins" not in cache
    assert "leaves the call" in agent


def test_clap_times_are_per_file():
    assert clap_times([1.25, 4.5]) == {"01.mov": [1.25, 4.5], "02.mov": []}

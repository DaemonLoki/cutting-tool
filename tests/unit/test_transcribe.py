"""Word merge, sentence indexes, and timestamp checks. The model is never loaded."""

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cutter.cli import app
from cutter.config import load_profile
from cutter.models import (
    Source,
    SourcesArtifact,
    SourcesData,
    Word,
    WordsArtifact,
    WordsData,
    make_meta,
    write_artifact,
)
from cutter.transcribe import (
    STAGE_VERSION,
    RawWord,
    Subword,
    TimestampError,
    assemble_words,
    merge_subwords,
    transcribe_project,
    words_between,
)


def test_merge_subwords_on_leading_space_and_sentencepiece_marker():
    tokens = [
        Subword("▁Hel", 0.00, 0.10, 1.0),
        Subword("lo", 0.10, 0.20, 1.0),
        Subword(",", 0.20, 0.28, 1.0),
        Subword(" the", 0.28, 0.40, 1.0),
        Subword(" ag", 0.40, 0.55, 0.25),
        Subword("ent", 0.55, 0.70, 1.0),
    ]
    words = merge_subwords(tokens)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hello,", 0.00, 0.28),
        ("the", 0.28, 0.40),
        ("agent", 0.40, 0.70),
    ]
    assert words[0].conf == pytest.approx(1.0)
    assert words[2].conf == pytest.approx(0.5)


def test_punctuation_stays_on_the_word_and_leaves_norm():
    raw = merge_subwords(
        [
            Subword(" So", 0.0, 0.2),
            Subword(",", 0.2, 0.3),
            Subword(" room", 0.3, 0.5),
            Subword(" 2", 0.5, 0.7),
            Subword(".", 0.7, 0.8),
        ]
    )
    data = assemble_words([("s01", 10.0, [*raw, RawWord("ﬁle,", 1.0, 1.2)])])
    assert [(word.w, word.norm) for word in data.words] == [
        ("So,", "so"),
        ("room", "room"),
        ("2.", "2"),
        ("ﬁle,", "file"),
    ]


def test_sentence_index_advances_after_punctuation_and_at_a_source_boundary():
    data = assemble_words(
        [
            (
                "s01",
                10.0,
                [
                    RawWord("Hello.", 0.0, 0.2),
                    RawWord("Really?", 0.3, 0.5),
                    RawWord("Yes!", 0.6, 0.8),
                    RawWord("and", 0.9, 1.1),
                ],
            ),
            ("s02", 5.0, [RawWord("then", 0.0, 0.2), RawWord("stop.", 0.3, 0.5)]),
        ]
    )
    assert [(word.i, word.source, word.w, word.sent) for word in data.words] == [
        (0, "s01", "Hello.", 0),
        (1, "s01", "Really?", 1),
        (2, "s01", "Yes!", 2),
        (3, "s01", "and", 3),
        (4, "s02", "then", 4),
        (5, "s02", "stop.", 4),
    ]


def test_source_boundary_does_not_advance_twice_after_terminal_punctuation():
    data = assemble_words(
        [
            ("s01", 10.0, [RawWord("Done.", 0.0, 0.2)]),
            ("s02", 5.0, [RawWord("Next", 0.0, 0.2)]),
        ]
    )
    assert [(word.i, word.source, word.sent) for word in data.words] == [
        (0, "s01", 0),
        (1, "s02", 1),
    ]


def test_tiny_timestamp_violations_are_clamped(caplog: pytest.LogCaptureFixture):
    raw = [
        RawWord("early", -0.02, 0.10),
        RawWord("one", 0.20, 0.40),
        RawWord("back", 0.18, 0.55),
        RawWord("tight", 0.60, 0.57),
        RawWord("tail", 0.90, 1.03),
    ]
    with caplog.at_level(logging.WARNING, logger="cutter.transcribe"):
        data = assemble_words([("s01", 1.0, raw)])
    assert [(word.w, word.start, word.end) for word in data.words] == [
        ("early", 0.0, 0.10),
        ("one", 0.20, 0.40),
        ("back", 0.20, 0.55),
        ("tight", 0.60, 0.60),
        ("tail", 0.90, 1.0),
    ]
    assert "clamped" in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        [RawWord("a", -0.05, 0.1)],
        [RawWord("a", 0.2, 0.3), RawWord("b", 0.15, 0.4)],
        [RawWord("a", 0.5, 0.45)],
        [RawWord("a", 0.9, 1.05)],
    ],
)
def test_timestamp_violation_of_50ms_raises(raw: list[RawWord]):
    with pytest.raises(TimestampError):
        assemble_words([("s01", 1.0, raw)])


def test_words_between_formats_words_whose_start_is_in_range():
    words = [
        Word(i=0, source="s01", w="So,", norm="so", start=1.2, end=1.38, conf=None, sent=0),
        Word(i=1, source="s01", w="there", norm="there", start=1.4, end=1.6, sent=0),
        Word(i=2, source="s01", w="now.", norm="now", start=2.0, end=2.2, sent=0),
    ]
    assert words_between(words, 1.2, 2.0) == "So, 1.200 1.380\nthere 1.400 1.600"


def test_transcribe_project_writes_words_and_skips_when_cached(tmp_path: Path):
    project = tmp_path / "proj"
    wav = project / "artifacts" / "audio" / "s01.16k.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF")
    write_artifact(
        project / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=1,
                inputs_hash="sha256:abc",
                config_hash="sha256:def",
            ),
            data=SourcesData(
                fps="30000/1001",
                width=1920,
                height=1080,
                audio_rate=48000,
                audio_channels=1,
                sources=[
                    Source(
                        id="s01",
                        path=str(project / "raw" / "01.mov"),
                        duration_s=2.0,
                        duration_frames=60,
                        start_timecode="00:00:00:00",
                        start_frames=0,
                        vfr_warning=False,
                        asr_wav="artifacts/audio/s01.16k.wav",
                        analysis_wav="artifacts/audio/s01.48k.wav",
                    )
                ],
            ),
        ),
    )
    calls: list[Path] = []

    class Fake:
        def transcribe(self, wav_path: Path) -> list[RawWord]:
            calls.append(wav_path)
            return [RawWord("Hello.", 0.1, 0.4, 0.9)]

    profile = load_profile("long")
    artifact = transcribe_project(project, profile=profile, transcriber=Fake())
    assert artifact.meta.stage == "transcribe"
    assert artifact.meta.stage_version == STAGE_VERSION
    assert artifact.data.words[0].w == "Hello."
    assert artifact.data.words[0].norm == "hello"
    assert artifact.data.words[0].sent == 0
    assert (project / "artifacts" / "words.json").is_file()

    again = transcribe_project(project, profile=profile, transcriber=Fake())
    assert again.data.words[0].w == "Hello."
    assert calls == [wav]

    forced = transcribe_project(project, profile=profile, force=True, transcriber=Fake())
    assert forced.data.words[0].i == 0
    assert len(calls) == 2


def test_cli_transcribe_without_sources_exits_1(tmp_path: Path):
    result = CliRunner().invoke(app, ["transcribe", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing sources" in result.output


def test_cli_words_prints_the_window(tmp_path: Path):
    project = tmp_path / "proj"
    write_artifact(
        project / "artifacts" / "words.json",
        WordsArtifact(
            meta=make_meta(
                stage="transcribe",
                stage_version=1,
                inputs_hash="sha256:abc",
                config_hash="sha256:def",
            ),
            data=WordsData(
                words=[
                    Word(
                        i=0,
                        source="s01",
                        w="So,",
                        norm="so",
                        start=1.2,
                        end=1.38,
                        conf=None,
                        sent=0,
                    )
                ]
            ),
        ),
    )
    result = CliRunner().invoke(app, ["words", str(project), "--from", "1", "--to", "2"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "So, 1.200 1.380"

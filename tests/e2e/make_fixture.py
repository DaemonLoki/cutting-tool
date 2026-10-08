"""Build a short project with deliberate retakes using ``say`` and ffmpeg.

The first source repeats one sentence, then aborts another. The second source
starts by finishing that aborted sentence. ``expected.txt`` is the transcript
that should remain after those failed takes are dropped.

``fillers``, ``clap``, and ``script`` add Phase 2 material. All three default
off, and that path writes the same project as before.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FILE_ONE = (
    "The cache stores the result and",
    "The cache stores the result and skips the work.",
    "The agent joins the call and",
)
FILE_TWO = ("The agent joins the call and subscribes to the audio track.",)
EXPECTED = (
    "The cache stores the result and skips the work. "
    "The agent joins the call and subscribes to the audio track."
)

_GAP_S = "0.3"
_CLAP_S = 0.03
_CLAP_AT_S = 0.2
_TRAILING_SILENCE_S = 1.0
_FILLER_LINE = "Um, the cache stores the result and, uh, skips the work."


def file_one_lines(*, fillers: bool = False) -> tuple[str, ...]:
    """Lines spoken in the first source. Fillers sit in the kept sentence."""
    second = _FILLER_LINE if fillers else FILE_ONE[1]
    return (FILE_ONE[0], second, FILE_ONE[2])


def script_markdown() -> str:
    """``script.md``: one heading per expected sentence, plus one unspoken line."""
    return (
        "# Cache\n"
        "\n"
        "The cache stores the result and skips the work.\n"
        "\n"
        "# Agent\n"
        "\n"
        "The agent joins the call and subscribes to the audio track.\n"
        "\n"
        "The agent leaves the call when the host ends it.\n"
    )


def clap_times(onsets_s: list[float]) -> dict[str, list[float]]:
    """Clap onsets in seconds, keyed by the raw file name."""
    return {"01.mov": [round(onset, 6) for onset in onsets_s], "02.mov": []}


def build_project(
    project_dir: Path,
    *,
    fillers: bool = False,
    clap: bool = False,
    script: bool = False,
) -> str:
    """Write ``raw/01.mov``, ``raw/02.mov``, and ``expected.txt``. Return the kept text.

    ``fillers`` puts um/uh into the kept sentence. ``expected.txt`` stays clean.
    ``clap`` replaces the gap before that retake with a 30 ms burst, and adds
    another burst inside a 1 s silence after the last line. ``script`` writes
    ``script.md``. Onset times go to ``claps.json`` beside ``expected.txt``.
    """
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("say and ffmpeg must be on PATH")
    project_dir = Path(project_dir)
    raw = project_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    work = project_dir / "_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    first = work / "one.wav"
    onsets = _speech(
        file_one_lines(fillers=fillers),
        first,
        clap_after=frozenset({0}) if clap else frozenset(),
    )
    # The end of a file is often transcribed with a period. Cut the last
    # word off so the failed take stays aborted. The trailing clap silence
    # is added after that cut, so the last word is still removed.
    _drop_tail(first, 0.80)
    if clap:
        onsets.append(_append_clap_in_silence(first))
    _mux(first, raw / "01.mov")
    second = work / "two.wav"
    _speech(FILE_TWO, second)
    _mux(second, raw / "02.mov")
    shutil.rmtree(work)
    (project_dir / "expected.txt").write_text(EXPECTED + "\n", encoding="utf-8")
    if clap:
        (project_dir / "claps.json").write_text(
            json.dumps(clap_times(onsets), indent=2) + "\n",
            encoding="utf-8",
        )
    if script:
        (project_dir / "script.md").write_text(script_markdown(), encoding="utf-8")
    return EXPECTED


def _speech(
    lines: tuple[str, ...],
    dest: Path,
    *,
    clap_after: frozenset[int] = frozenset(),
) -> list[float]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / f"{dest.name}-parts"
    work.mkdir()
    parts: list[Path] = []
    onsets: list[float] = []
    cursor = 0.0
    for index, line in enumerate(lines):
        wav = work / f"{index}.wav"
        _say(line, wav)
        parts.append(wav)
        if clap_after:
            cursor += _duration(wav)
        if index == len(lines) - 1:
            continue
        if index in clap_after:
            burst = work / f"{index}-clap.wav"
            _clap(burst)
            parts.append(burst)
            onsets.append(cursor)
            cursor += _duration(burst)
            continue
        silence = work / f"{index}-gap.wav"
        _silence(silence)
        parts.append(silence)
        if clap_after:
            cursor += _duration(silence)
    _concat(parts, dest)
    return onsets


def _say(text: str, wav: Path) -> None:
    aiff = wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(aiff),
            "-ar",
            "48000",
            "-ac",
            "1",
            str(wav),
        ],
        check=True,
    )


def _silence(dest: Path, seconds: str = _GAP_S) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            seconds,
            str(dest),
        ],
        check=True,
    )


def _clap(dest: Path) -> None:
    """30 ms white noise with a 25 ms fade-out starting at 5 ms."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=d=0.03:c=white:a=0.8:r=48000",
            "-af",
            "afade=t=out:st=0.005:d=0.025",
            "-t",
            "0.03",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
    )


def _append_clap_in_silence(wav: Path) -> float:
    """Append 1 s of silence with a clap 200 ms in. Return the onset in ``wav``."""
    before = _duration(wav)
    work = wav.parent / f"{wav.name}-tail"
    work.mkdir()
    prefix = work / "prefix.wav"
    burst = work / "clap.wav"
    suffix = work / "suffix.wav"
    _silence(prefix, f"{_CLAP_AT_S:.3f}")
    _clap(burst)
    rest = _TRAILING_SILENCE_S - _CLAP_AT_S - _CLAP_S
    _silence(suffix, f"{rest:.3f}")
    combined = work / "combined.wav"
    _concat([wav, prefix, burst, suffix], combined)
    onset = before + _duration(prefix)
    combined.replace(wav)
    shutil.rmtree(work)
    return onset


def _duration(wav: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(wav),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


def _concat(parts: list[Path], dest: Path) -> None:
    args = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    for part in parts:
        args.extend(["-i", str(part)])
    graph = "".join(f"[{index}:a]" for index in range(len(parts)))
    graph += f"concat=n={len(parts)}:v=0:a=1[a]"
    args.extend(["-filter_complex", graph, "-map", "[a]", str(dest)])
    subprocess.run(args, check=True)


def _drop_tail(wav: Path, seconds: float) -> None:
    """Remove the last ``seconds`` so the final word is not in the file."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(wav),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    keep = max(0.1, float(probe.stdout.strip()) - seconds)
    trimmed = wav.with_name(f"{wav.stem}-cut.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(wav),
            "-t",
            f"{keep:.3f}",
            "-c:a",
            "pcm_s16le",
            str(trimmed),
        ],
        check=True,
    )
    trimmed.replace(wav)


def _mux(speech: Path, dest: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30000/1001",
            "-i",
            str(speech),
            "-shortest",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "0",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        check=True,
    )

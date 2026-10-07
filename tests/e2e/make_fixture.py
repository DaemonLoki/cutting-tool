"""Build a short project with deliberate retakes using ``say`` and ffmpeg.

The first source repeats one sentence, then aborts another. The second source
starts by finishing that aborted sentence. ``expected.txt`` is the transcript
that should remain after those failed takes are dropped.
"""

from __future__ import annotations

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


def build_project(project_dir: Path) -> str:
    """Write ``raw/01.mov``, ``raw/02.mov``, and ``expected.txt``. Return the kept text."""
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("say and ffmpeg must be on PATH")
    project_dir = Path(project_dir)
    raw = project_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    work = project_dir / "_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    first = _speech(FILE_ONE, work / "one.wav")
    # The end of a file is often transcribed with a period. Cut the last
    # word off so the failed take stays aborted.
    _drop_tail(first, 0.80)
    _mux(first, raw / "01.mov")
    _mux(_speech(FILE_TWO, work / "two.wav"), raw / "02.mov")
    shutil.rmtree(work)
    (project_dir / "expected.txt").write_text(EXPECTED + "\n", encoding="utf-8")
    return EXPECTED


def _speech(lines: tuple[str, ...], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = dest.parent / f"{dest.name}-parts"
    work.mkdir()
    parts: list[Path] = []
    for index, line in enumerate(lines):
        wav = work / f"{index}.wav"
        _say(line, wav)
        parts.append(wav)
        if index != len(lines) - 1:
            silence = work / f"{index}-gap.wav"
            _silence(silence)
            parts.append(silence)
    _concat(parts, dest)
    return dest


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


def _silence(dest: Path) -> None:
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
            _GAP_S,
            str(dest),
        ],
        check=True,
    )


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

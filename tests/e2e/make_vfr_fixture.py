"""Build a synthetic variable-frame-rate source for the T6 drift check.

Short ``testsrc2`` segments alternate 30 fps and 24 fps. The long project
puts a ``say -v Samantha`` sentence at the start of each segment, one every
10 seconds, for about 3 minutes, and writes ``raw/01.mp4``.

``-fps_mode`` is an output option (``ffmpeg -h full`` lists it; ``passthrough``
is accepted and an unknown mode is rejected). The concat *filter* with
``-fps_mode passthrough`` keeps both frame durations, so ``r_frame_rate`` and
``avg_frame_rate`` differ. The concat demuxer with ``-c copy -fps_mode
passthrough`` does not: a 2 s + 2 s pair collapses onto ~30 fps timestamps
(``avg_frame_rate`` still differs, but the 24 fps half is not really 24 fps).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

_SIZE = "640x360"
_RATES = (30, 24)
_CADENCE_S = 10
_TOTAL_S = 180
_TINY_SEGMENT_S = 2


def build_project(
    project_dir: Path,
    *,
    cadence_s: int = _CADENCE_S,
    total_s: int = _TOTAL_S,
) -> Path:
    """Write ``raw/01.mp4`` of about ``total_s`` seconds. Return that path.

    A sentence starts each segment. ``total_s`` is rounded down to a whole
    number of ``cadence_s`` segments.
    """
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("say and ffmpeg must be on PATH")
    if cadence_s <= 0 or total_s < cadence_s:
        raise ValueError("cadence_s and total_s must leave at least one segment")
    count = total_s // cadence_s
    segments = [
        (_RATES[index % 2], cadence_s, _sentence(index, index * cadence_s))
        for index in range(count)
    ]
    return _build(Path(project_dir), segments)


def build_tiny(project_dir: Path, *, segment_s: int = _TINY_SEGMENT_S) -> Path:
    """Two segments of a few seconds, with a generated tone instead of ``say``.

    This is the file the default test ingests. It is variable frame rate on
    the same concat path as the long source.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg must be on PATH")
    if segment_s <= 0:
        raise ValueError("segment_s must be positive")
    segments = [(_RATES[0], segment_s, None), (_RATES[1], segment_s, None)]
    return _build(Path(project_dir), segments)


def probe_video(path: Path) -> tuple[str, str, float]:
    """Return ``r_frame_rate``, ``avg_frame_rate``, and duration in seconds."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    video = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
    duration = float(payload["format"]["duration"])
    return str(video["r_frame_rate"]), str(video["avg_frame_rate"]), duration


def _sentence(index: int, at_s: int) -> str:
    return f"This is marker {index + 1} at {at_s} seconds."


def _build(
    project_dir: Path,
    segments: list[tuple[int, int, str | None]],
) -> Path:
    raw = project_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    work = project_dir / "_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    parts: list[Path] = []
    try:
        for index, (rate, duration_s, speech) in enumerate(segments):
            audio: Path | None = None
            if speech is not None:
                audio = work / f"{index}.wav"
                _say(speech, audio)
            part = work / f"seg-{index}.mp4"
            _encode_segment(part, rate=rate, duration_s=duration_s, audio=audio)
            parts.append(part)
        dest = raw / "01.mp4"
        _concat_passthrough(parts, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
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


def _encode_segment(dest: Path, *, rate: int, duration_s: int, audio: Path | None) -> None:
    duration = str(duration_s)
    args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={_SIZE}:rate={rate}:duration={duration}",
    ]
    if audio is None:
        args.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:duration={duration}",
            ]
        )
    else:
        args.extend(["-i", str(audio)])
    args.extend(
        [
            "-t",
            duration,
            "-af",
            "apad",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(dest),
        ]
    )
    subprocess.run(args, check=True)


def _concat_passthrough(parts: list[Path], dest: Path) -> None:
    args = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    for part in parts:
        args.extend(["-i", str(part)])
    pairs = "".join(f"[{index}:v][{index}:a]" for index in range(len(parts)))
    graph = f"{pairs}concat=n={len(parts)}:v=1:a=1[v][a]"
    args.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(dest),
        ]
    )
    subprocess.run(args, check=True)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m tests.e2e.make_vfr_fixture <project-dir>")
    dest = build_project(Path(args[0]))
    r_rate, avg_rate, duration = probe_video(dest)
    print(f"{dest} duration={duration:.3f}s r_frame_rate={r_rate} avg_frame_rate={avg_rate}")


if __name__ == "__main__":
    main()

"""Validate a project's sources and extract audio. Video is never re-encoded."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from natsort import natsorted

from cutter.cache import STAGE_CONFIG_SECTIONS, cache_hit, config_hash, inputs_hash
from cutter.config import Profile
from cutter.models import Source, SourcesArtifact, SourcesData, make_meta, write_artifact

STAGE_VERSION = 1

Command = Callable[[list[str]], str]


class IngestError(Exception):
    """A source failed validation. The CLI exits 1."""


class ToolMissing(Exception):
    """ffmpeg or ffprobe is not on PATH. The CLI exits 2."""


@dataclass(frozen=True)
class MediaInfo:
    """What ffprobe reported for one source, in exact fractions where it matters."""

    path: Path
    r_frame_rate: Fraction
    avg_frame_rate: Fraction
    width: int
    height: int
    audio_rate: int
    audio_channels: int
    timecode: str
    duration: Fraction

    @property
    def nominal_fps(self) -> int:
        """30 for 30000/1001, 24 for 24000/1001, otherwise the rounded rate."""
        return round(self.r_frame_rate)

    @property
    def start_frames(self) -> int:
        return timecode_to_frames(self.timecode, self.nominal_fps)

    @property
    def duration_s(self) -> float:
        return float(self.duration)

    @property
    def duration_frames(self) -> int:
        return math.floor(self.duration * self.r_frame_rate)

    @property
    def vfr_warning(self) -> bool:
        return self.avg_frame_rate != self.r_frame_rate


def list_sources(raw_dir: Path, allowed_extensions: list[str]) -> list[Path]:
    """Numbered sources in the raw folder, hidden files ignored, natural order.

    ``10.mov`` sorts after ``2.mov``. ``.MP4`` matches ``.mp4``.
    """
    if not raw_dir.is_dir():
        raise IngestError(f"no raw folder at {raw_dir}")
    allowed = {ext.casefold() for ext in allowed_extensions}
    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.casefold() in allowed
    ]
    if not files:
        raise IngestError(f"no sources in {raw_dir}")
    return natsorted(files, key=lambda path: path.name)


def timecode_to_frames(timecode: str, nominal_fps: int) -> int:
    """Non-drop-frame ``HH:MM:SS:FF`` at the nominal rate, which is 30 for 29.97."""
    if ";" in timecode:
        raise IngestError("drop-frame timecode not supported in Phase 1")
    parts = timecode.split(":")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        raise IngestError(f"timecode {timecode!r} is not HH:MM:SS:FF")
    hours, minutes, seconds, frames = (int(part) for part in parts)
    return ((hours * 3600 + minutes * 60 + seconds) * nominal_fps) + frames


def probe_file(path: Path, *, command: Command | None = None) -> MediaInfo:
    """Read one source with ffprobe. Duration prefers the exact tick count."""
    run = command or _default_command
    raw = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path.name}: ffprobe did not return JSON") from exc
    try:
        return _parse_probe(path, payload)
    except IngestError as exc:
        raise IngestError(f"{path.name}: {exc}") from exc


def ingest_project(
    project_dir: Path,
    profile: Profile,
    *,
    force: bool = False,
    command: Command | None = None,
) -> SourcesData:
    """Probe ``raw/``, extract 16 kHz and 48 kHz audio, and write ``sources.json``."""
    run = command or _default_command
    if command is None:
        _require_tools()
    sources = list_sources(project_dir / "raw", profile.ingest.allowed_extensions)
    artifact_path = project_dir / "artifacts" / "sources.json"
    media_hash = inputs_hash(media=sources)
    cfg_hash = config_hash(profile, STAGE_CONFIG_SECTIONS["ingest"])
    if not force and cache_hit(
        artifact_path,
        stage="ingest",
        stage_version=STAGE_VERSION,
        inputs_hash=media_hash,
        config_hash=cfg_hash,
    ):
        cached = SourcesArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        if _wavs_present(project_dir, cached.data):
            return cached.data

    infos = [probe_file(path, command=run) for path in sources]
    _require_same_shape(infos)
    for info in infos:
        if info.vfr_warning:
            print(
                f"VFR warning: {info.path.name} avg_frame_rate {info.avg_frame_rate} "
                f"!= r_frame_rate {info.r_frame_rate}",
                file=sys.stderr,
            )
    data = _sources_data(infos)
    for info, source in zip(infos, data.sources, strict=True):
        _extract_wav(run, info.path, project_dir / source.asr_wav, profile.ingest.asr_sample_rate)
        _extract_wav(
            run,
            info.path,
            project_dir / source.analysis_wav,
            profile.ingest.analysis_sample_rate,
        )
    write_artifact(
        artifact_path,
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=STAGE_VERSION,
                inputs_hash=media_hash,
                config_hash=cfg_hash,
            ),
            data=data,
        ),
    )
    return data


def _parse_probe(path: Path, payload: dict) -> MediaInfo:
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None:
        raise IngestError("has no video stream")
    if audio is None:
        raise IngestError("has no audio stream")
    rate = _fraction(video.get("r_frame_rate"), "frame rate")
    if rate <= 0:
        raise IngestError("has no frame rate")
    avg_raw = video.get("avg_frame_rate")
    average = _fraction(avg_raw, "avg frame rate") if avg_raw not in (None, "0/0") else rate
    sample_rate = audio.get("sample_rate")
    channels = audio.get("channels")
    if sample_rate is None or channels is None:
        raise IngestError("audio stream is missing sample rate or channels")
    width = video.get("width")
    height = video.get("height")
    if width is None or height is None:
        raise IngestError("video stream is missing width or height")
    duration = _duration(video) or _duration(payload.get("format") or {})
    if duration is None or duration <= 0:
        raise IngestError("has no duration")
    timecode = _timecode(streams, payload.get("format") or {})
    timecode_to_frames(timecode, round(rate))
    return MediaInfo(
        path=path,
        r_frame_rate=rate,
        avg_frame_rate=average,
        width=int(width),
        height=int(height),
        audio_rate=int(sample_rate),
        audio_channels=int(channels),
        timecode=timecode,
        duration=duration,
    )


def _duration(stream: dict) -> Fraction | None:
    """Prefer duration ticks. ffprobe's decimal duration can round a frame away."""
    ticks = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if ticks not in (None, "N/A") and time_base not in (None, "0/0"):
        numerator, _, denominator = str(time_base).partition("/")
        if denominator and int(denominator) != 0 and int(ticks) > 0:
            return Fraction(int(ticks), int(denominator))
    raw = stream.get("duration")
    if raw in (None, "N/A"):
        return None
    return Fraction(str(raw))


def _timecode(streams: list[dict], fmt: dict) -> str:
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    from_video = (video.get("tags") or {}).get("timecode")
    if from_video:
        return str(from_video)
    for stream in streams:
        if str(stream.get("codec_tag_string", "")).casefold() != "tmcd":
            continue
        from_track = (stream.get("tags") or {}).get("timecode")
        if from_track:
            return str(from_track)
    from_format = (fmt.get("tags") or {}).get("timecode")
    if from_format:
        return str(from_format)
    return "00:00:00:00"


def _require_same_shape(infos: list[MediaInfo]) -> None:
    first = infos[0]
    for info in infos[1:]:
        if info.r_frame_rate != first.r_frame_rate:
            raise IngestError(
                f"{info.path.name} frame rate {info.r_frame_rate} does not match "
                f"{first.path.name} ({first.r_frame_rate})"
            )
        if info.width != first.width or info.height != first.height:
            raise IngestError(
                f"{info.path.name} frame size {info.width}x{info.height} does not match "
                f"{first.path.name} ({first.width}x{first.height})"
            )
        if info.audio_rate != first.audio_rate:
            raise IngestError(
                f"{info.path.name} audio sample rate {info.audio_rate} does not match "
                f"{first.path.name} ({first.audio_rate})"
            )


def _sources_data(infos: list[MediaInfo]) -> SourcesData:
    first = infos[0]
    sources: list[Source] = []
    for index, info in enumerate(infos, start=1):
        source_id = f"s{index:02d}"
        sources.append(
            Source(
                id=source_id,
                path=str(info.path.resolve()),
                duration_s=info.duration_s,
                duration_frames=info.duration_frames,
                start_timecode=info.timecode,
                start_frames=info.start_frames,
                vfr_warning=info.vfr_warning,
                asr_wav=f"artifacts/audio/{source_id}.16k.wav",
                analysis_wav=f"artifacts/audio/{source_id}.48k.wav",
            )
        )
    return SourcesData(
        fps=f"{first.r_frame_rate.numerator}/{first.r_frame_rate.denominator}",
        width=first.width,
        height=first.height,
        audio_rate=first.audio_rate,
        audio_channels=first.audio_channels,
        sources=sources,
    )


def _extract_wav(run: Command, source: Path, dest: Path, sample_rate: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def _wavs_present(project_dir: Path, data: SourcesData) -> bool:
    return all(
        (project_dir / source.asr_wav).is_file() and (project_dir / source.analysis_wav).is_file()
        for source in data.sources
    )


def _fraction(raw: object, label: str) -> Fraction:
    if raw in (None, "N/A"):
        raise IngestError(f"missing {label}")
    try:
        return Fraction(str(raw))
    except (ValueError, ZeroDivisionError) as exc:
        raise IngestError(f"invalid {label}: {raw}") from exc


def _require_tools() -> None:
    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            raise ToolMissing(f"{tool} is not on PATH")


def _default_command(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ToolMissing(f"{cmd[0]} is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"{cmd[0]} failed"
        raise IngestError(detail) from exc
    return completed.stdout

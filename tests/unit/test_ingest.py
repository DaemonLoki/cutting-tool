"""Ingest: source order, shared geometry, timecode, and audio extraction."""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cutter.cli import app
from cutter.config import load_profile
from cutter.ingest import (
    IngestError,
    ingest_project,
    list_sources,
    probe_file,
    timecode_to_frames,
)

runner = CliRunner()
REPO = Path(__file__).resolve().parents[2]
SAMPLE_RAW = REPO / "test" / "fixtures" / "sample" / "raw"


def _ffprobe(
    *,
    rate: str = "30000/1001",
    average: str | None = None,
    width: int = 1920,
    height: int = 1080,
    duration: str = "10",
    duration_ts: int | None = None,
    time_base: str = "1/30000",
    sample_rate: str | None = "48000",
    channels: int | None = 2,
    timecode: str | None = "00:00:00:00",
    video_timecode: str | None = None,
    audio: bool = True,
) -> dict:
    video: dict = {
        "codec_type": "video",
        "r_frame_rate": rate,
        "avg_frame_rate": average or rate,
        "width": width,
        "height": height,
        "duration": duration,
    }
    if duration_ts is not None:
        video["duration_ts"] = duration_ts
        video["time_base"] = time_base
    if video_timecode is not None:
        video["tags"] = {"timecode": video_timecode}
    streams: list[dict] = [video]
    if audio:
        audio_stream: dict = {"codec_type": "audio"}
        if sample_rate is not None:
            audio_stream["sample_rate"] = sample_rate
        if channels is not None:
            audio_stream["channels"] = channels
        streams.append(audio_stream)
    fmt: dict = {"duration": duration, "tags": {}}
    if timecode is not None and video_timecode is None:
        streams.append(
            {
                "codec_type": "data",
                "codec_tag_string": "tmcd",
                "tags": {"timecode": timecode},
            }
        )
        fmt["tags"]["timecode"] = timecode
    return {"streams": streams, "format": fmt}


def _command(payloads: dict[str, dict]):
    calls: list[list[str]] = []

    def command(cmd: list[str]) -> str:
        calls.append(list(cmd))
        if cmd[0] == "ffprobe":
            return json.dumps(payloads[Path(cmd[-1]).name])
        if cmd[0] == "ffmpeg":
            dest = Path(cmd[-1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"RIFF")
            return ""
        raise AssertionError(cmd)

    command.calls = calls  # type: ignore[attr-defined]
    return command


def _touch(raw: Path, *names: str) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in names:
        (raw / name).write_bytes(b"source")


def test_ten_sorts_after_two_and_uppercase_extension_is_accepted(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw, "10.mov", "2.mov", "clip.MP4", ".hidden.mov", "notes.txt")
    names = [path.name for path in list_sources(raw, [".mov", ".mp4", ".m4v"])]
    assert names == ["2.mov", "10.mov", "clip.MP4"]


def test_timecode_one_hour_at_29_97_ndf_is_108000_frames():
    assert timecode_to_frames("01:00:00:00", 30) == 108000
    with pytest.raises(IngestError, match="drop-frame timecode not supported in Phase 1"):
        timecode_to_frames("01:00:00;00", 30)


def test_probe_uses_nominal_30_for_29_97(tmp_path: Path):
    path = tmp_path / "hour.mov"
    path.write_bytes(b"x")
    info = probe_file(
        path,
        command=_command({"hour.mov": _ffprobe(rate="30000/1001", timecode="01:00:00:00")}),
    )
    assert info.nominal_fps == 30
    assert info.start_frames == 108000
    assert info.duration_frames == 299


def test_mismatched_frame_rate_raises(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw, "2.mov", "10.mov")
    command = _command(
        {
            "2.mov": _ffprobe(rate="30000/1001"),
            "10.mov": _ffprobe(rate="24000/1001"),
        }
    )
    with pytest.raises(IngestError, match="frame rate"):
        ingest_project(tmp_path, load_profile(), command=command)


def test_no_audio_and_audio_rate_mismatch(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw, "a.mov")
    with pytest.raises(IngestError, match="no audio stream"):
        ingest_project(
            tmp_path,
            load_profile(),
            command=_command({"a.mov": _ffprobe(audio=False)}),
        )
    _touch(raw, "b.mov")
    with pytest.raises(IngestError, match="audio sample rate"):
        ingest_project(
            tmp_path,
            load_profile(),
            command=_command(
                {
                    "a.mov": _ffprobe(sample_rate="48000"),
                    "b.mov": _ffprobe(sample_rate="44100"),
                }
            ),
        )


def test_drop_frame_timecode_raises(tmp_path: Path):
    _touch(tmp_path / "raw", "a.mov")
    with pytest.raises(IngestError, match="drop-frame timecode not supported in Phase 1"):
        ingest_project(
            tmp_path,
            load_profile(),
            command=_command({"a.mov": _ffprobe(timecode="01:00:00;00")}),
        )


def test_vfr_warning_is_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _touch(tmp_path / "raw", "2.mov")
    data = ingest_project(
        tmp_path,
        load_profile(),
        command=_command({"2.mov": _ffprobe(rate="30000/1001", average="30/1")}),
    )
    err = capsys.readouterr().err
    assert data.sources[0].vfr_warning is True
    assert "VFR warning" in err
    assert "2.mov" in err


def test_exact_ticks_keep_the_last_frame(tmp_path: Path):
    """ffprobe's decimal duration rounds 928 frames down; ticks do not."""
    path = tmp_path / "intro.MP4"
    path.write_bytes(b"x")
    info = probe_file(
        path,
        command=_command(
            {
                "intro.MP4": _ffprobe(
                    rate="24000/1001",
                    duration="38.705333",
                    duration_ts=928928,
                    time_base="1/24000",
                    timecode="01:20:27:14",
                )
            }
        ),
    )
    assert info.duration_frames == 928
    assert info.nominal_fps == 24
    assert info.start_frames == (1 * 3600 + 20 * 60 + 27) * 24 + 14


def test_ingest_writes_wavs_and_cache_skips_the_second_run(tmp_path: Path):
    _touch(tmp_path / "raw", "2.mov", "10.mov")
    payload = _ffprobe(rate="24000/1001", timecode="01:00:00:00")
    command = _command({"2.mov": payload, "10.mov": payload})
    data = ingest_project(tmp_path, load_profile(), command=command)
    assert [source.id for source in data.sources] == ["s01", "s02"]
    assert [Path(source.path).name for source in data.sources] == ["2.mov", "10.mov"]
    assert data.fps == "24000/1001"
    assert data.sources[0].start_frames == 3600 * 24
    assert (tmp_path / "artifacts" / "audio" / "s01.16k.wav").is_file()
    assert (tmp_path / "artifacts" / "audio" / "s02.48k.wav").is_file()
    ffmpeg_calls = [cmd for cmd in command.calls if cmd[0] == "ffmpeg"]
    assert ffmpeg_calls[0][ffmpeg_calls[0].index("-map") : -1] == [
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    ingest_project(tmp_path, load_profile(), command=command)
    assert [cmd for cmd in command.calls if cmd[0] == "ffmpeg"] == ffmpeg_calls


def test_force_extracts_again(tmp_path: Path):
    _touch(tmp_path / "raw", "2.mov")
    command = _command({"2.mov": _ffprobe()})
    ingest_project(tmp_path, load_profile(), command=command)
    before = len(command.calls)
    ingest_project(tmp_path, load_profile(), force=True, command=command)
    assert len(command.calls) > before


def test_cli_validation_and_missing_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 1
    assert "no raw folder" in result.output

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.mov").write_bytes(b"x")
    monkeypatch.setattr("cutter.ingest.shutil.which", lambda _name: None)
    missing = runner.invoke(app, ["ingest", str(tmp_path)])
    assert missing.exit_code == 2
    assert "not on PATH" in missing.output


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")
@pytest.mark.skipif(not (SAMPLE_RAW / "1-intro.MP4").is_file(), reason="sample sources are absent")
def test_sample_sources_match_the_camera():
    intro = probe_file(SAMPLE_RAW / "1-intro.MP4")
    skills = probe_file(SAMPLE_RAW / "2-skills.MP4")
    for info in (intro, skills):
        assert info.width == 3840
        assert info.height == 2160
        assert info.r_frame_rate == Fraction(24000, 1001)
        assert info.audio_rate == 48000
        assert info.audio_channels == 2
        assert info.vfr_warning is False
        assert ";" not in info.timecode
    assert intro.timecode == "01:20:27:14"
    assert intro.start_frames == 115862
    assert intro.duration_frames == 928
    assert skills.timecode == "01:19:28:13"
    assert skills.start_frames == 114445
    assert skills.duration_frames == 1417
    names = [path.name for path in list_sources(SAMPLE_RAW, [".mov", ".mp4", ".m4v"])]
    assert names == ["1-intro.MP4", "2-skills.MP4"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_ffmpeg_round_trip_keeps_timecode(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_clip(raw / "2.mov", rate="30000/1001", timecode="01:00:00:00")
    _write_clip(raw / "10.mov", rate="30000/1001", timecode="01:00:00:00")
    data = ingest_project(tmp_path, load_profile())
    assert [Path(source.path).name for source in data.sources] == ["2.mov", "10.mov"]
    assert data.sources[0].start_frames == 108000
    assert data.fps == "30000/1001"
    assert (tmp_path / "artifacts" / "audio" / "s01.16k.wav").stat().st_size > 0


def _write_clip(path: Path, *, rate: str, timecode: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x240:rate={rate}:duration=0.3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.3",
            "-shortest",
            "-timecode",
            timecode,
            "-c:v",
            "mpeg4",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )

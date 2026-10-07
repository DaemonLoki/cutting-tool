"""Profile follows the displayed frame unless --profile is set."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cutter.choose import FrameUnknown, choose_profile, oriented_size, profile_name_for_frame
from cutter.cli import app
from cutter.ingest import probe_file
from cutter.models import Source, SourcesArtifact, SourcesData, make_meta, write_artifact

runner = CliRunner()


def test_profile_follows_the_displayed_frame():
    assert profile_name_for_frame(3840, 2160) == "long"
    assert profile_name_for_frame(1080, 1080) == "long"
    assert profile_name_for_frame(1080, 1920) == "short"
    assert oriented_size(1920, 1080, rotation=-90) == (1080, 1920)
    assert oriented_size(1920, 1080, rotation=180) == (1920, 1080)


def test_sideways_storage_selects_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    clip = raw / "1.mov"
    clip.write_bytes(b"not a movie")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "width": 1920,
                "height": 1080,
                "duration": "1",
                "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
            },
            {"codec_type": "audio", "sample_rate": "48000", "channels": 2},
        ],
        "format": {"duration": "1"},
    }

    def command(cmd: list[str]) -> str:
        assert cmd[0] == "ffprobe"
        return json.dumps(payload)

    info = probe_file(clip, command=command)
    assert info.rotation == -90
    assert (info.width, info.height) == (1920, 1080)

    monkeypatch.setattr("cutter.choose._require_tools", lambda: None)
    monkeypatch.setattr("cutter.choose.probe_file", lambda _path: info)
    choice = choose_profile(tmp_path)
    assert choice.name == "short"
    assert choice.orientation == "vertical"
    assert (choice.width, choice.height) == (1080, 1920)


def test_unknown_frame_when_nothing_to_measure(tmp_path: Path):
    with pytest.raises(FrameUnknown):
        choose_profile(tmp_path)


def test_sources_artifact_selects_short_without_raw(tmp_path: Path):
    _write_sources(tmp_path, width=1080, height=1920)
    choice = choose_profile(tmp_path)
    assert choice.name == "short"
    assert choice.orientation == "vertical"
    assert (choice.width, choice.height) == (1080, 1920)


def test_run_announces_the_automatic_profile(tmp_path: Path):
    _write_sources(tmp_path, width=1080, height=1920)
    result = runner.invoke(app, ["run", str(tmp_path)])
    assert result.exit_code == 1
    assert "profile short (vertical, 1080x1920)" in result.stdout
    assert "no raw folder" in result.output

    forced = runner.invoke(app, ["run", str(tmp_path), "--profile", "long"])
    assert "profile short" not in forced.stdout
    assert forced.exit_code == 1


def _write_sources(project_dir: Path, *, width: int, height: int) -> None:
    data = SourcesData(
        fps="30000/1001",
        width=width,
        height=height,
        audio_rate=48000,
        audio_channels=2,
        sources=[
            Source(
                id="s01",
                path=str(project_dir / "raw" / "1.mov"),
                duration_s=1.0,
                duration_frames=30,
                start_timecode="00:00:00:00",
                start_frames=0,
                vfr_warning=False,
                asr_wav="artifacts/audio/s01.16k.wav",
                analysis_wav="artifacts/audio/s01.48k.wav",
            )
        ],
    )
    write_artifact(
        project_dir / "artifacts" / "sources.json",
        SourcesArtifact(
            meta=make_meta(
                stage="ingest",
                stage_version=1,
                inputs_hash="sha256:sources",
                config_hash="sha256:cfg",
            ),
            data=data,
        ),
    )

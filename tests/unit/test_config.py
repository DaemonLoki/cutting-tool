"""Profile loading and --set overrides."""

from pathlib import Path

import pytest
import yaml

from cutter.config import PROFILES_DIR, ConfigError, load_profile

DTD = (
    "/Applications/Final Cut Pro.app/Contents/Frameworks/"
    "Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd"
)


def test_short_profile_cuts_tighter_than_long():
    long_form = load_profile("long")
    short_form = load_profile("short")
    assert short_form.tighten.max_gap_ms < long_form.tighten.max_gap_ms
    assert short_form.tighten.pad_head_ms < long_form.tighten.pad_head_ms
    assert short_form.tighten.pad_tail_ms < long_form.tighten.pad_tail_ms
    assert short_form.tighten.snap_window_ms < long_form.tighten.snap_window_ms
    assert short_form.tighten.min_head_ms < long_form.tighten.min_head_ms
    assert short_form.tighten.min_tail_ms < long_form.tighten.min_tail_ms
    assert short_form.tighten.pad_head_ms >= short_form.tighten.min_head_ms
    assert short_form.tighten.pad_tail_ms >= short_form.tighten.min_tail_ms
    assert short_form.retakes == long_form.retakes
    assert short_form.judge == long_form.judge


def test_long_profile_loads():
    profile = load_profile("long")
    assert profile.fcpxml.version == "1.14"
    assert profile.fcpxml.dtd_path == DTD
    assert profile.ingest.allowed_extensions == [".mov", ".mp4", ".m4v"]
    assert profile.retakes.stopwords_file is None
    assert profile.judge.enabled is True
    assert profile.judge.min_confidence == 0.7
    assert profile.tighten.max_gap_ms == 400


def test_profile_rejects_unknown_keys(tmp_path: Path):
    data = yaml.safe_load((PROFILES_DIR / "long.yaml").read_text(encoding="utf-8"))
    data["nope"] = 1
    (tmp_path / "long.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="nope"):
        load_profile("long", profiles_dir=tmp_path)

    data.pop("nope")
    data["ingest"]["nope"] = 1
    (tmp_path / "long.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="nope"):
        load_profile("long", profiles_dir=tmp_path)


def test_set_overrides_single_values():
    profile = load_profile(
        "long",
        [
            "judge.enabled=false",
            "fcpxml.version=1.14",
            "retakes.missing_content_ratio=0.5",
            "retakes.auto_drop_max_s=30",
            "ingest.allowed_extensions=['.mov']",
            "retakes.stopwords_file=null",
        ],
    )
    assert profile.judge.enabled is False
    assert profile.fcpxml.version == "1.14"
    assert profile.retakes.missing_content_ratio == 0.5
    assert profile.retakes.auto_drop_max_s == 30
    assert profile.ingest.allowed_extensions == [".mov"]
    assert profile.retakes.stopwords_file is None


def test_set_can_point_stopwords_at_a_file():
    profile = load_profile("long", ["retakes.stopwords_file=words.txt"])
    assert profile.retakes.stopwords_file == "words.txt"


def test_set_rejects_unknown_paths_and_bad_values():
    with pytest.raises(ConfigError, match="unknown config path"):
        load_profile("long", ["ingest.nope=1"])
    with pytest.raises(ConfigError, match="invalid value"):
        load_profile("long", ["judge.enabled=maybe"])
    with pytest.raises(ConfigError, match="key.path=value"):
        load_profile("long", ["judge.enabled"])
    with pytest.raises(ConfigError, match="single value"):
        load_profile("long", ["ingest=1"])


def test_phase2_sections_match_long_unless_documented():
    long_form = load_profile("long")
    short_form = load_profile("short")
    assert long_form.ingest.cfr_proxy == "never"
    assert long_form.ingest.proxy_codec == "prores_proxy"
    assert long_form.vad.enabled is True
    assert long_form.vad.backend == "silero"
    assert long_form.vad.threshold == 0.5
    assert long_form.claps.min_match_words == 2
    assert long_form.fillers.words[0] == "um"
    assert long_form.fillers.phrases == []
    assert long_form.script.path == "script.md"
    assert long_form.script.prefer == "best"
    assert long_form.chapters.levels == [1, 2]
    assert long_form.tighten.use_vad is True
    assert long_form.fcpxml.rejects_include_fillers is False
    assert long_form.fcpxml.vfr_media == "original"
    assert short_form.vad == long_form.vad
    assert short_form.claps == long_form.claps
    assert short_form.fillers == long_form.fillers
    assert short_form.script == long_form.script
    assert short_form.chapters == long_form.chapters
    assert short_form.ingest.cfr_proxy == long_form.ingest.cfr_proxy
    assert short_form.ingest.proxy_codec == long_form.ingest.proxy_codec
    assert short_form.tighten.use_vad == long_form.tighten.use_vad
    assert short_form.fcpxml.rejects_include_fillers == long_form.fcpxml.rejects_include_fillers
    assert short_form.fcpxml.vfr_media == long_form.fcpxml.vfr_media


def test_set_overrides_phase2_keys_including_lists():
    profile = load_profile(
        "long",
        [
            "ingest.cfr_proxy=auto",
            "ingest.proxy_codec=h264",
            "vad.threshold=0.2",
            "tighten.use_vad=false",
            "fcpxml.vfr_media=proxy",
            "fcpxml.rejects_include_fillers=true",
            "chapters.levels=[1]",
            "fillers.words=['um', 'uh']",
            "script.prefer=last",
        ],
    )
    assert profile.ingest.cfr_proxy == "auto"
    assert profile.ingest.proxy_codec == "h264"
    assert profile.vad.threshold == 0.2
    assert profile.tighten.use_vad is False
    assert profile.fcpxml.vfr_media == "proxy"
    assert profile.fcpxml.rejects_include_fillers is True
    assert profile.chapters.levels == [1]
    assert profile.fillers.words == ["um", "uh"]
    assert profile.script.prefer == "last"


def test_set_rejects_an_unknown_proxy_mode():
    with pytest.raises(ConfigError, match="cfr_proxy"):
        load_profile("long", ["ingest.cfr_proxy=sometimes"])


def test_missing_profile():
    with pytest.raises(ConfigError, match="not found"):
        load_profile("missing")

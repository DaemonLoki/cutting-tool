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


def test_missing_profile():
    with pytest.raises(ConfigError, match="not found"):
        load_profile("missing")

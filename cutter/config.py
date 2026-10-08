"""Named recording profiles and ``--set`` overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = REPO_ROOT / "profiles"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestConfig(StrictModel):
    allowed_extensions: list[str]
    asr_sample_rate: int
    analysis_sample_rate: int
    cfr_proxy: Literal["never", "auto", "always"]
    proxy_codec: Literal["prores_proxy", "h264"]


class TranscribeConfig(StrictModel):
    model: str
    chunk_duration_s: int
    overlap_duration_s: int


class RetakesConfig(StrictModel):
    window_words: int
    min_match_words: int
    word_similarity: int = Field(ge=0, le=100)
    max_mismatches: int
    max_lookback_s: int
    auto_drop_max_s: int
    missing_content_ratio: float
    stopwords_file: str | None


class JudgeConfig(StrictModel):
    enabled: bool
    base_url: str
    model: str
    min_confidence: float = Field(ge=0, le=1)
    timeout_s: int


class VadConfig(StrictModel):
    enabled: bool
    backend: Literal["silero", "energy"]
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    pad_ms: int
    flag_unheard_speech_s: float


class ClapsConfig(StrictModel):
    enabled: bool
    min_rise_db: float
    max_duration_ms: int
    min_gap_ms: int
    min_match_words: int
    exclude_before_ms: int
    exclude_after_ms: int


class FillersConfig(StrictModel):
    enabled: bool
    words: list[str]
    phrases: list[str]
    sentence_start_words: list[str]
    max_duration_s: float


class ScriptConfig(StrictModel):
    path: str
    min_take_score: int
    prefer: Literal["best", "last"]
    min_score_gap: int
    drop_later_takes: bool
    flag_unscripted_s: float


class ChaptersConfig(StrictModel):
    enabled: bool
    levels: list[int]


class TightenConfig(StrictModel):
    max_gap_ms: int
    pad_head_ms: int
    pad_tail_ms: int
    snap_window_ms: int
    min_head_ms: int
    min_tail_ms: int
    rms_frame_ms: int
    voice_margin_db: float
    voice_quiet_ms: int
    min_range_frames: int
    use_vad: bool


class FcpxmlConfig(StrictModel):
    version: str
    dtd_path: str
    event_name: str
    project_name: str
    rejects_project_name: str
    audio_role: str
    rejects_include_fillers: bool
    vfr_media: Literal["original", "proxy"]


class Profile(StrictModel):
    ingest: IngestConfig
    transcribe: TranscribeConfig
    retakes: RetakesConfig
    judge: JudgeConfig
    vad: VadConfig
    claps: ClapsConfig
    fillers: FillersConfig
    script: ScriptConfig
    chapters: ChaptersConfig
    tighten: TightenConfig
    fcpxml: FcpxmlConfig


class ConfigError(Exception):
    """A profile file or ``--set`` override could not be applied."""


def load_profile(
    name: str = "long",
    overrides: list[str] | None = None,
    *,
    profiles_dir: Path | None = None,
) -> Profile:
    """Load ``profiles/<name>.yaml`` and apply ``key.path=value`` overrides."""
    directory = PROFILES_DIR if profiles_dir is None else profiles_dir
    path = directory / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"profile {name!r} not found at {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"profile {path} is not valid YAML") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"profile {path} must be a mapping")
    data: dict[str, Any] = loaded
    for override in overrides or []:
        _apply_override(data, override)
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override {override!r} must be key.path=value")
    key, raw = override.split("=", 1)
    parts = key.split(".")
    if not key or any(part == "" for part in parts):
        raise ConfigError(f"override {override!r} has an empty path")
    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or not isinstance(cursor.get(part), dict):
            raise ConfigError(f"unknown config path: {key}")
        cursor = cursor[part]
    leaf = parts[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise ConfigError(f"unknown config path: {key}")
    current = cursor[leaf]
    if isinstance(current, dict):
        raise ConfigError(f"override {key} must name a single value")
    cursor[leaf] = _coerce(raw, current, key)


def _coerce(raw: str, current: Any, key: str) -> Any:
    if isinstance(current, str) or current is None:
        if raw == "null":
            return None
        return raw
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid value for {key}: {raw!r}") from exc
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"invalid value for {key}: {raw!r}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"invalid value for {key}: {raw!r}")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"invalid value for {key}: {raw!r}")
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ConfigError(f"invalid value for {key}: {raw!r}")
        return value
    raise ConfigError(f"cannot override {key}")

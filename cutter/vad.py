"""Speech segments from Silero or from the analysis-wav loudness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pysilero_vad import SileroVoiceActivityDetector

from cutter.audio import rms_envelope, voice_levels
from cutter.config import VadConfig

_SILERO_RATE = 16_000
_SILERO_SAMPLES = 512
_SILERO_BYTES = _SILERO_SAMPLES * 2
_ENERGY_FRAME_MS = 10


def speech_segments(
    wav_path: Path,
    config: VadConfig,
    *,
    voice_margin_db: float,
) -> list[tuple[float, float]]:
    """Return sorted, non-overlapping ``(start, end)`` spans in seconds.

    ``silero`` reads a mono 16 kHz WAV. ``energy`` reads the analysis WAV
    (48 kHz) and marks a frame as speech when its smoothed level is at least
    ``voice_margin_db`` above that file's background. The caller does not
    call this when ``vad.enabled`` is false.
    """
    stored, _gate = speech_and_clap_gate(wav_path, config, voice_margin_db=voice_margin_db)
    return stored


def speech_and_clap_gate(
    wav_path: Path,
    config: VadConfig,
    *,
    voice_margin_db: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Stored speech segments, and the spans that hide a clap.

    Stored segments bridge pauses shorter than ``min_silence_ms`` and pad
    both ends. The clap gate does neither. A mistake mark sits in the pause
    between utterances, and that pause is often shorter than the bridge.
    Only a voiced run at least ``min_speech_ms`` long is a gate, so a clap
    that the detector marks as a short blip is not treated as speech.
    """
    if config.backend == "silero":
        voiced, frame_s, duration = _silero_mask(wav_path, config.threshold)
    else:
        voiced, frame_s, duration = _energy_mask(wav_path, voice_margin_db)
    spans = _mask_to_spans(voiced, frame_s)
    min_speech_s = config.min_speech_ms / 1000
    gate = [(start, end) for start, end in spans if end - start >= min_speech_s]
    return _postprocess(spans, config, duration), gate


def _silero_mask(wav_path: Path, threshold: float) -> tuple[np.ndarray, float, float]:
    """One boolean per 512-sample chunk. A short tail is padded with silence."""
    audio, sample_rate = sf.read(wav_path, dtype="int16", always_2d=False)
    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise ValueError(f"silero expects mono audio: {wav_path}")
    if int(sample_rate) != _SILERO_RATE:
        raise ValueError(f"silero expects 16 kHz audio, got {sample_rate} Hz: {wav_path}")
    duration = samples.size / _SILERO_RATE
    frame_s = _SILERO_SAMPLES / _SILERO_RATE
    if samples.size == 0:
        return np.empty(0, dtype=bool), frame_s, duration

    detector = SileroVoiceActivityDetector()
    detector.reset()
    raw = np.ascontiguousarray(samples, dtype=np.int16).tobytes()
    voiced: list[bool] = []
    for offset in range(0, len(raw), _SILERO_BYTES):
        piece = raw[offset : offset + _SILERO_BYTES]
        # process_chunk raises InvalidChunkSizeError unless the chunk is 1024 bytes.
        if len(piece) != _SILERO_BYTES:
            piece = piece.ljust(_SILERO_BYTES, b"\x00")
        voiced.append(float(detector.process_chunk(piece)) >= threshold)
    return np.asarray(voiced, dtype=bool), frame_s, duration


def _energy_mask(wav_path: Path, voice_margin_db: float) -> tuple[np.ndarray, float, float]:
    envelope, _sample_rate = rms_envelope(wav_path, _ENERGY_FRAME_MS)
    levels, background = voice_levels(envelope, _ENERGY_FRAME_MS)
    frame_s = _ENERGY_FRAME_MS / 1000
    duration = float(sf.info(wav_path).duration)
    if levels.size == 0:
        return np.empty(0, dtype=bool), frame_s, duration
    return levels >= background + voice_margin_db, frame_s, duration


def _mask_to_spans(voiced: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for index, is_speech in enumerate(voiced.tolist()):
        if is_speech and start is None:
            start = index
        elif not is_speech and start is not None:
            spans.append((start * frame_s, index * frame_s))
            start = None
    if start is not None:
        spans.append((start * frame_s, len(voiced) * frame_s))
    return spans


def _postprocess(
    spans: list[tuple[float, float]],
    config: VadConfig,
    duration: float,
) -> list[tuple[float, float]]:
    bridged = _bridge(spans, config.min_silence_ms / 1000)
    min_speech_s = config.min_speech_ms / 1000
    kept = [(start, end) for start, end in bridged if end - start >= min_speech_s]
    pad_s = config.pad_ms / 1000
    padded: list[tuple[float, float]] = []
    for start, end in kept:
        start = max(0.0, start - pad_s)
        end = min(duration, end + pad_s)
        if start < end:
            padded.append((float(start), float(end)))
    return _merge(padded)


def _bridge(
    spans: list[tuple[float, float]],
    min_silence_s: float,
) -> list[tuple[float, float]]:
    if not spans:
        return []
    bridged: list[tuple[float, float]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = bridged[-1]
        if start - prev_end < min_silence_s:
            bridged[-1] = (prev_start, end)
        else:
            bridged.append((start, end))
    return bridged


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    merged: list[tuple[float, float]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

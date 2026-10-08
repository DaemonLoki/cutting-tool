"""RMS energy envelopes and silence snapping for cut points."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

_SMOOTH_MS = 50


def rms_envelope(wav_path: Path, frame_ms: int) -> tuple[np.ndarray, int]:
    """RMS energy per frame of a mono WAV.

    Returns ``(envelope, sample_rate)``. Frame ``i`` covers
    ``[i * frame_ms, (i + 1) * frame_ms)`` milliseconds. The last frame may be
    shorter. Empty audio returns an empty float64 array. Multi-channel input
    is averaged to mono. RMS is ``sqrt(mean(square))``; a silent frame is 0.
    """
    audio, sample_rate = sf.read(wav_path, always_2d=False)
    rate = int(sample_rate)
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        return np.empty(0, dtype=np.float64), rate

    span = frame_ms * rate
    total = int(samples.size)
    n_frames = (total * 1000 + span - 1) // span
    envelope = np.empty(n_frames, dtype=np.float64)
    for index in range(n_frames):
        start = index * span // 1000
        end = min(total, (index + 1) * span // 1000)
        frame = samples[start:end]
        if frame.size == 0:
            envelope[index] = 0.0
        else:
            envelope[index] = math.sqrt(float(np.mean(frame * frame)))
    return envelope, rate


def voice_levels(envelope: np.ndarray, frame_ms: int) -> tuple[np.ndarray, float]:
    """Smoothed levels in dB, and the background level of the whole source.

    Each level is the RMS averaged over ``_SMOOTH_MS``, so one loud or quiet
    frame does not decide. The background is the 10th percentile of those
    levels. An empty envelope returns an empty array and ``-inf``.
    """
    if envelope.size == 0:
        return np.empty(0, dtype=np.float64), float("-inf")
    width = max(1, round(_SMOOTH_MS / frame_ms))
    smoothed = np.convolve(envelope, np.ones(width) / width, mode="same")
    levels = 20 * np.log10(np.maximum(smoothed, 1e-10))
    return levels, float(np.percentile(levels, 10))


def voice_end(
    levels: np.ndarray,
    *,
    frame_ms: int,
    start_s: float,
    end_s: float,
    threshold_db: float,
    quiet_ms: int,
) -> float | None:
    """Where the voice inside ``[start_s, end_s]`` stops, in seconds.

    ``levels`` come from ``voice_levels``. A frame is voiced at or above
    ``threshold_db``. The voice ends where a run of unvoiced frames,
    ``quiet_ms`` long, starts after a voiced frame and before ``end_s``. The
    run may continue past ``end_s``. Returns None when the word has no voiced
    frame, or when no such run starts inside the word.
    """
    total = int(levels.shape[0])
    if total == 0 or end_s <= start_s:
        return None
    step = frame_ms / 1000
    first = max(0, math.floor(start_s / step))
    last = min(total - 1, math.ceil(end_s / step) - 1)
    if first > last:
        return None
    need = max(1, math.ceil(quiet_ms / frame_ms))

    voiced = False
    run_start: int | None = None
    for index in range(first, min(total, last + need + 1)):
        if float(levels[index]) >= threshold_db:
            if index > last:
                return None
            voiced = True
            run_start = None
            continue
        if not voiced:
            if index > last:
                return None
            continue
        if run_start is None:
            if index > last:
                return None
            run_start = index
        if index - run_start + 1 >= need:
            return run_start * step
    return None


def snap_time(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: int,
    raw_s: float,
    window_ms: int,
    earliest_s: float,
    latest_s: float,
    forbidden: Sequence[tuple[float, float]] | None = None,
) -> float:
    """Pick the lowest-RMS time inside the snap window and ``[earliest_s, latest_s]``.

    Search times are frame centers. Consider frames whose center lies in
    ``[raw_s - window_ms/1000, raw_s + window_ms/1000]`` and in
    ``[earliest_s, latest_s]``. Choose the minimum RMS. Ties go to the center
    closest to ``raw_s``, then the earlier one. If ``earliest_s > latest_s``,
    or no frame center lies in the intersection, return ``raw_s``.

    ``forbidden`` intervals are clap zones in seconds. A frame that overlaps
    one is skipped. If every candidate frame is forbidden, return ``raw_s``.
    """
    if earliest_s > latest_s or envelope.size == 0:
        return raw_s
    half = window_ms / 1000
    low = max(raw_s - half, earliest_s)
    high = min(raw_s + half, latest_s)
    if low > high:
        return raw_s

    step = frame_ms / 1000
    best_index: int | None = None
    best_rms = 0.0
    best_distance = 0.0
    for index in range(int(envelope.shape[0])):
        frame_start = index * step
        frame_end = (index + 1) * step
        center = (index + 0.5) * step
        if center < low or center > high:
            continue
        if _frame_forbidden(frame_start, frame_end, forbidden):
            continue
        energy = float(envelope[index])
        distance = abs(center - raw_s)
        closer = best_index is None or energy < best_rms
        tied = (
            best_index is not None
            and energy == best_rms
            and (distance < best_distance or (distance == best_distance and index < best_index))
        )
        if closer or tied:
            best_index = index
            best_rms = energy
            best_distance = distance
    if best_index is None:
        return raw_s
    return (best_index + 0.5) * step


def _frame_forbidden(
    frame_start: float,
    frame_end: float,
    forbidden: Sequence[tuple[float, float]] | None,
) -> bool:
    """True when this RMS frame overlaps a forbidden interval.

    Touching an endpoint is allowed: the frame may end as the zone starts.
    """
    if not forbidden:
        return False
    return any(
        frame_start < zone_end and frame_end > zone_start for zone_start, zone_end in forbidden
    )

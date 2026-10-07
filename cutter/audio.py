"""RMS energy envelopes and silence snapping for cut points."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


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


def snap_time(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: int,
    raw_s: float,
    window_ms: int,
    earliest_s: float,
    latest_s: float,
) -> float:
    """Pick the lowest-RMS time inside the snap window and ``[earliest_s, latest_s]``.

    Search times are frame centers. Consider frames whose center lies in
    ``[raw_s - window_ms/1000, raw_s + window_ms/1000]`` and in
    ``[earliest_s, latest_s]``. Choose the minimum RMS. Ties go to the center
    closest to ``raw_s``, then the earlier one. If ``earliest_s > latest_s``,
    or no frame center lies in the intersection, return ``raw_s``.
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
        center = (index + 0.5) * step
        if center < low or center > high:
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

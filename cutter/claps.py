"""Short broadband transients on a 48 kHz analysis WAV.

The background is a 1-second rolling median of the 2 ms envelope. Each frame
updates that median in logarithmic time, so a long recording does not sort
the window again from scratch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

from cutter.audio import rms_envelope
from cutter.config import ClapsConfig

_FRAME_MS = 2
_BACKGROUND_MS = 1000
_RETURN_DB = 6.0
_LEVEL_FLOOR = 1e-10


@dataclass(frozen=True)
class DetectedClap:
    """One clap, without a source id. ``t`` is the onset in seconds."""

    t: float
    peak_db: float
    rise_db: float


def detect_claps(
    wav_48k: Path,
    speech: list[tuple[float, float]],
    config: ClapsConfig,
) -> list[DetectedClap]:
    """Clap onsets in ``wav_48k`` that fall outside ``speech``.

    ``speech`` is inclusive on both ends. An empty list keeps every onset.
    ``config.enabled`` is ignored here; the caller skips the call when claps
    are off. Events closer than ``min_gap_ms`` collapse into the earliest
    onset, keeping the louder peak and that peak's rise.
    """
    envelope, _rate = rms_envelope(wav_48k, _FRAME_MS)
    if envelope.size < 4:
        return []
    level = 20 * np.log10(np.maximum(envelope, _LEVEL_FLOOR))
    window = _BACKGROUND_MS // _FRAME_MS
    background = _rolling_median(level, window)
    frame_s = _FRAME_MS / 1000
    min_rise = config.min_rise_db
    quick_rise = min_rise / 2
    found: list[DetectedClap] = []
    last = int(level.shape[0])
    # The background is a 1000 ms median. Until that window is full, a leading
    # silence makes the first speech attack look like a short clap.
    first = max(3, window - 1)
    for index in range(first, last):
        if level[index] - background[index] < min_rise:
            continue
        if level[index] - level[index - 3] < quick_rise:
            continue
        onset = index * frame_s
        if _inside_speech(onset, speech):
            continue
        end = index + 1
        while end < last and level[end] > background[end] + _RETURN_DB:
            end += 1
        duration_ms = (end - index) * _FRAME_MS
        if duration_ms > config.max_duration_ms:
            continue
        peak_db = float(np.max(level[index:end]))
        found.append(
            DetectedClap(
                t=onset,
                peak_db=peak_db,
                rise_db=peak_db - float(background[index]),
            )
        )
    return _merge(found, config.min_gap_ms / 1000)


def _inside_speech(onset: float, speech: list[tuple[float, float]]) -> bool:
    return any(start <= onset <= end for start, end in speech)


def _merge(events: list[DetectedClap], min_gap_s: float) -> list[DetectedClap]:
    """Collapse a chain of onsets whose consecutive gaps are under ``min_gap_s``."""
    if not events:
        return []
    merged = [events[0]]
    previous = events[0].t
    for event in events[1:]:
        if event.t - previous < min_gap_s:
            kept = merged[-1]
            if event.peak_db > kept.peak_db:
                merged[-1] = DetectedClap(t=kept.t, peak_db=event.peak_db, rise_db=event.rise_db)
            previous = event.t
            continue
        merged.append(event)
        previous = event.t
    return merged


def _rolling_median(level: np.ndarray, window: int) -> np.ndarray:
    """Median of the trailing ``window`` samples, including the current one.

    The first frames use however many samples exist so far. An even window
    averages the two central values, matching ``numpy.median``.
    """
    count = int(level.shape[0])
    background = np.empty(count, dtype=np.float64)
    tracker = _WindowMedian(window)
    for index in range(count):
        tracker.push(float(level[index]))
        background[index] = tracker.median()
    return background


class _WindowMedian:
    """Sliding-window median. Each push is O(log window), including duplicates."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        self._window = window
        self._low: list[tuple[float, int]] = []
        self._high: list[tuple[float, int]] = []
        self._dead: set[int] = set()
        self._side: dict[int, str] = {}
        self._low_size = 0
        self._high_size = 0
        self._order: deque[int] = deque()
        self._seq = 0

    def push(self, value: float) -> None:
        seq = self._seq
        self._seq += 1
        self._order.append(seq)
        self._add(value, seq)
        if len(self._order) > self._window:
            self._remove(self._order.popleft())

    def median(self) -> float:
        self._prune(self._low)
        if self._low_size > self._high_size:
            return -self._low[0][0]
        self._prune(self._high)
        return (-self._low[0][0] + self._high[0][0]) / 2

    def _add(self, value: float, seq: int) -> None:
        self._prune(self._low)
        if self._low_size == 0 or value <= -self._low[0][0]:
            heappush(self._low, (-value, seq))
            self._side[seq] = "low"
            self._low_size += 1
        else:
            heappush(self._high, (value, seq))
            self._side[seq] = "high"
            self._high_size += 1
        self._rebalance()

    def _remove(self, seq: int) -> None:
        self._dead.add(seq)
        if self._side.pop(seq) == "low":
            self._low_size -= 1
        else:
            self._high_size -= 1
        self._prune(self._low)
        self._prune(self._high)
        self._rebalance()

    def _rebalance(self) -> None:
        while self._low_size > self._high_size + 1:
            self._prune(self._low)
            value, seq = heappop(self._low)
            self._low_size -= 1
            heappush(self._high, (-value, seq))
            self._side[seq] = "high"
            self._high_size += 1
        while self._high_size > self._low_size:
            self._prune(self._high)
            value, seq = heappop(self._high)
            self._high_size -= 1
            heappush(self._low, (-value, seq))
            self._side[seq] = "low"
            self._low_size += 1

    def _prune(self, heap: list[tuple[float, int]]) -> None:
        while heap and heap[0][1] in self._dead:
            _, seq = heappop(heap)
            self._dead.discard(seq)

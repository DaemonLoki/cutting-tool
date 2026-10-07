"""RMS envelopes and silence snapping."""

from pathlib import Path

import numpy as np
import soundfile as sf

from cutter.audio import rms_envelope, snap_time


def test_rms_envelope_peaks_on_a_middle_spike(tmp_path: Path):
    sample_rate = 48_000
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[24_000:24_100] = 0.8
    path = tmp_path / "spike.wav"
    sf.write(path, samples, sample_rate)

    envelope, rate = rms_envelope(path, 10)

    assert rate == sample_rate
    assert envelope.shape == (100,)
    assert envelope.dtype == np.float64
    assert envelope[0] == 0
    assert envelope[-1] == 0
    assert envelope[50] > 0.1
    assert envelope[50] == max(envelope)


def test_snap_time_chooses_the_quiet_frame_inside_the_window():
    envelope = np.array([1.0, 0.2, 1.0, 0.2, 1.0], dtype=np.float64)
    snapped = snap_time(
        envelope,
        sample_rate=48_000,
        frame_ms=10,
        raw_s=0.020,
        window_ms=30,
        earliest_s=0.0,
        latest_s=1.0,
    )
    assert snapped == (1 + 0.5) * 10 / 1000


def test_snap_time_respects_earliest_and_latest():
    envelope = np.array([1.0, 1.0, 0.0, 0.4, 1.0], dtype=np.float64)
    snapped = snap_time(
        envelope,
        sample_rate=48_000,
        frame_ms=10,
        raw_s=0.030,
        window_ms=40,
        earliest_s=0.030,
        latest_s=0.050,
    )
    assert snapped == (3 + 0.5) * 10 / 1000


def test_snap_time_returns_raw_when_the_legal_interval_is_empty():
    envelope = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    snapped = snap_time(
        envelope,
        sample_rate=48_000,
        frame_ms=10,
        raw_s=0.2,
        window_ms=50,
        earliest_s=1.0,
        latest_s=0.4,
    )
    assert snapped == 0.2

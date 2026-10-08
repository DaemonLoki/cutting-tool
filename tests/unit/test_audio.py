"""RMS envelopes and silence snapping."""

from pathlib import Path

import numpy as np
import soundfile as sf

from cutter.audio import rms_envelope, snap_time, voice_end, voice_levels

QUIET = 0.001  # -60 dB
LOUD = 0.1  # -20 dB


def _levels(*runs: tuple[float, int]) -> tuple[np.ndarray, float]:
    """10 ms frames: each run is (rms, frame count)."""
    envelope = np.concatenate([np.full(count, rms) for rms, count in runs])
    return voice_levels(envelope, 10)


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


def test_voice_levels_background_is_the_quiet_tenth():
    levels, background = _levels((QUIET, 200), (LOUD, 100), (QUIET, 200))
    assert levels.shape == (500,)
    assert round(background) == -60


def test_voice_end_finds_silence_inside_a_stretched_word():
    # Voice from 1.0 s to 1.3 s. The transcript says the word lasts to 2.0 s.
    levels, background = _levels((QUIET, 100), (LOUD, 30), (QUIET, 170))
    end = voice_end(
        levels,
        frame_ms=10,
        start_s=1.0,
        end_s=2.0,
        threshold_db=background + 12,
        quiet_ms=300,
    )
    assert end is not None
    assert 1.3 <= end < 1.34


def test_voice_end_ignores_a_dip_shorter_than_quiet_ms():
    # 100 ms of quiet between two syllables is not the end of the word.
    levels, background = _levels(
        (QUIET, 100), (LOUD, 20), (QUIET, 10), (LOUD, 20), (QUIET, 150)
    )
    end = voice_end(
        levels,
        frame_ms=10,
        start_s=1.0,
        end_s=2.0,
        threshold_db=background + 12,
        quiet_ms=300,
    )
    assert end is not None
    assert end >= 1.5


def test_voice_end_none_without_voice_or_when_voice_continues():
    silent, background = _levels((QUIET, 300))
    assert (
        voice_end(
            silent,
            frame_ms=10,
            start_s=1.0,
            end_s=2.0,
            threshold_db=background + 12,
            quiet_ms=300,
        )
        is None
    )
    voiced, background = _levels((QUIET, 100), (LOUD, 100), (QUIET, 100))
    assert (
        voice_end(
            voiced,
            frame_ms=10,
            start_s=1.0,
            end_s=2.0,
            threshold_db=background + 12,
            quiet_ms=300,
        )
        is None
    )


def test_snap_time_skips_a_forbidden_quiet_frame():
    envelope = np.array([1.0, 0.0, 1.0, 0.2, 1.0], dtype=np.float64)
    snapped = snap_time(
        envelope,
        sample_rate=48_000,
        frame_ms=10,
        raw_s=0.015,
        window_ms=40,
        earliest_s=0.0,
        latest_s=1.0,
        forbidden=[(0.005, 0.025)],
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

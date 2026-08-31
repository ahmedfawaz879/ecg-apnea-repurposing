import numpy as np
import pytest

from src.features.hrv_features import extract_hrv_features

FS = 100
QRS_PULSE = np.array([0.2, 1.0, -0.3, 0.1, 0.05], dtype=np.float32)  # same impulse shape as
                                                                       # src/data/apnea_ecg_loader.py's
                                                                       # make_synthetic_apnea_ecg


def _make_qrs_like_signal(beat_times: np.ndarray, fs: int, duration_s: float) -> np.ndarray:
    n_samples = int(duration_s * fs)
    sig = np.zeros(n_samples, dtype=np.float32)
    for bt in beat_times:
        idx = int(round(bt * fs))
        if 0 <= idx < n_samples - len(QRS_PULSE):
            sig[idx:idx + len(QRS_PULSE)] += QRS_PULSE
    return sig


def test_hrv_features_sdnn_matches_hand_computed_sinusoidal_rr():
    """Builds a synthetic RR series that varies sinusoidally around a base
    interval (a known, hand-computable ground truth), synthesizes a
    QRS-like ECG segment whose beats land exactly on that RR series, and
    checks that extract_hrv_features's detector-derived mean_rr/sdnn match
    the ground-truth RR series' own mean/std to within one sample (1/FS)
    of R-peak-detection jitter."""
    n_beats = 60
    base_rr = 0.9        # seconds
    amplitude = 0.15      # seconds
    period_beats = 10

    beat_index = np.arange(n_beats)
    rr_intervals = base_rr + amplitude * np.sin(2 * np.pi * beat_index / period_beats)

    beat_times = np.concatenate([[0.0], np.cumsum(rr_intervals)])
    duration_s = beat_times[-1] + 2.0

    segment = _make_qrs_like_signal(beat_times, fs=FS, duration_s=duration_s)

    feats = extract_hrv_features(segment, fs=FS, min_beats_required=20)

    assert feats is not None

    expected_mean_rr = rr_intervals.mean()
    expected_sdnn = rr_intervals.std(ddof=1)

    tol = 1.0 / FS  # one sample of R-peak-detection jitter

    assert feats["mean_rr"] == pytest.approx(expected_mean_rr, abs=tol)
    assert feats["sdnn"] == pytest.approx(expected_sdnn, abs=tol)
    assert set(feats.keys()) == {
        "mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr",
        "lf_power", "hf_power", "lf_hf_ratio",
    }


def test_hrv_features_returns_none_below_min_beats_required():
    short_segment = _make_qrs_like_signal(np.array([0.0, 0.9]), fs=FS, duration_s=2.0)
    feats = extract_hrv_features(short_segment, fs=FS, min_beats_required=20)
    assert feats is None

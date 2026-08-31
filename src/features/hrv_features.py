"""Extracted unchanged from notebook Sec. 4 ("HRV Feature Extraction --
R-Peak Detection, Time- and Frequency-Domain").

Also hosts the StandardScaler fit/transform helpers factored out of
Sec. 5's Tier A/B training blocks, since both tiers share the same
scaler fit on TRAIN_FEATURES only (never refit on test data) -- kept
here, next to HRV_COLS, as the single shared location rather than
duplicated in both tier_a_logistic.py and tier_b_gbtree.py.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from wfdb.processing import xqrs_detect

from src.utils.config import Config


def extract_hrv_features(segment: np.ndarray, fs: int, min_beats_required: int) -> Optional[Dict[str, float]]:
    try:
        qrs_inds = xqrs_detect(sig=segment, fs=fs, verbose=False)
    except Exception:
        return None
    if len(qrs_inds) < min_beats_required:
        return None

    rr = np.diff(qrs_inds) / fs   # seconds
    rr = rr[(rr > 0.3) & (rr < 2.0)]   # physiological range (30-200 bpm); drops detector artifacts
    if len(rr) < min_beats_required - 1:
        return None

    mean_rr = rr.mean()
    sdnn = rr.std(ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50 = np.mean(np.abs(np.diff(rr)) > 0.05)
    mean_hr = 60.0 / mean_rr

    # Frequency domain: interpolate the (unevenly sampled) RR series onto a
    # uniform 4 Hz grid, then Welch PSD. LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz.
    beat_times = np.cumsum(rr)
    if beat_times[-1] < 10:   # too short a span for a meaningful spectrum
        lf_power, hf_power, lf_hf_ratio = np.nan, np.nan, np.nan
    else:
        fs_interp = 4.0
        t_uniform = np.arange(0, beat_times[-1], 1.0 / fs_interp)
        interp_fn = interp1d(beat_times, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")
        rr_uniform = interp_fn(t_uniform)
        freqs, psd = welch(rr_uniform, fs=fs_interp, nperseg=min(256, len(rr_uniform)))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else np.nan
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else np.nan
        lf_hf_ratio = lf_power / hf_power if (hf_power and hf_power > 0) else np.nan

    return {"mean_rr": mean_rr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50,
            "mean_hr": mean_hr, "lf_power": lf_power, "hf_power": hf_power, "lf_hf_ratio": lf_hf_ratio}


def build_feature_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feats = extract_hrv_features(row["raw_segment"], cfg.fs, cfg.min_beats_required)
        if feats is None:
            continue
        feats.update({"recording_id": row["recording_id"], "label": row["label"],
                       "center_minute_segment": row["center_minute_segment"]})
        rows.append(feats)
    return pd.DataFrame(rows)


HRV_COLS = ["mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "lf_power", "hf_power", "lf_hf_ratio"]


# ============================================================
# Shared scaler fit/transform (Sec. 5) -- fit on TRAIN_FEATURES only
# ============================================================

def fit_hrv_scaler(train_features: pd.DataFrame) -> Tuple[StandardScaler, pd.Series]:
    """Fits a StandardScaler on TRAIN_FEATURES[HRV_COLS] (median-imputed),
    and returns the train medians alongside it -- both are needed downstream
    since test data must be imputed with TRAIN medians, never its own."""
    train_medians = train_features[HRV_COLS].median()
    scaler = StandardScaler().fit(train_features[HRV_COLS].fillna(train_medians))
    return scaler, train_medians


def transform_hrv_features(df: pd.DataFrame, scaler: StandardScaler, train_medians: pd.Series) -> np.ndarray:
    X = df[HRV_COLS].fillna(train_medians)   # impute with TRAIN medians, never test's own
    return scaler.transform(X)

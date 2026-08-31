"""Unchanged from the earlier notebooks in this series -- dataset-agnostic."""
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_diagnostics(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    bin_acc, bin_conf, bin_count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_acc[b] = labels[mask].mean()
        bin_conf[b] = probs[mask].mean()
        bin_count[b] = mask.sum()
    ece = np.sum(bin_count / len(probs) * np.abs(bin_acc - bin_conf))
    brier = brier_score_loss(labels, probs)
    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    try:
        lr = LogisticRegression().fit(logits.reshape(-1, 1), labels)
        slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        slope, intercept = np.nan, np.nan
    return {"ece": ece, "brier": brier, "slope": slope, "intercept": intercept,
            "bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}

"""Unchanged from the earlier notebooks in this series -- dataset-agnostic."""
from typing import Tuple

import numpy as np


def fast_delong(labels: np.ndarray, preds_1: np.ndarray, preds_2: np.ndarray) -> Tuple[float, float]:
    def compute_midrank(x):
        order = np.argsort(x)
        ranked = np.empty(len(x))
        ranked[order] = np.arange(1, len(x) + 1)
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and x[order[j]] == x[order[j + 1]]:
                j += 1
            ranked[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return ranked

    def delong_components(preds, labels):
        pos = preds[labels == 1]; neg = preds[labels == 0]
        m, n = len(pos), len(neg)
        tx = compute_midrank(pos); ty = compute_midrank(neg)
        tz = compute_midrank(np.concatenate([pos, neg]))
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
        return auc, v01, v10

    labels = np.asarray(labels); preds_1 = np.asarray(preds_1); preds_2 = np.asarray(preds_2)
    auc1, v01_1, v10_1 = delong_components(preds_1, labels)
    auc2, v01_2, v10_2 = delong_components(preds_2, labels)
    v01 = np.vstack([v01_1, v01_2]); v10 = np.vstack([v10_1, v10_2])
    s01 = np.cov(v01); s10 = np.cov(v10)
    m, n = v01.shape[1], v10.shape[1]
    cov = s01 / m + s10 / n
    diff = auc1 - auc2
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return diff, np.nan
    z = diff / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return diff, p

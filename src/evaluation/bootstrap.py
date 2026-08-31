"""Unchanged from the earlier notebooks in this series -- dataset-agnostic."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42


def clustered_bootstrap_auroc(df: pd.DataFrame, label_col: str, pred_col: str,
                               cluster_col: str = "recording_id", n_boot: int = 500,
                               alpha: float = 0.05, seed: int = SEED) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    clusters = df[cluster_col].unique()
    point_auc = roc_auc_score(df[label_col], df[pred_col])
    boot_aucs = []
    for _ in range(n_boot):
        sample_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sample_df = df[df[cluster_col].isin(sample_clusters)]
        if sample_df[label_col].nunique() < 2:
            continue
        boot_aucs.append(roc_auc_score(sample_df[label_col], sample_df[pred_col]))
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auroc": point_auc, "ci_lo": lo, "ci_hi": hi, "n_boot_valid": len(boot_aucs)}

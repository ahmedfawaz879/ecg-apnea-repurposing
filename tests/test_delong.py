import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.delong import fast_delong


def test_fast_delong_diff_against_known_auc():
    """A constant (uninformative) predictor has AUROC == 0.5 exactly, so
    fast_delong's AUC-difference output must equal auc(preds_1) - 0.5,
    where auc(preds_1) is independently computed with sklearn -- a known,
    reproducible value against which fast_delong's own AUC computation
    can be checked without touching its internals."""
    rng = np.random.RandomState(0)
    n = 200
    labels = (rng.rand(n) < 0.4).astype(float)
    preds_1 = np.clip(labels * 0.6 + rng.normal(0, 0.25, n), 0, 1)
    preds_2 = np.full(n, 0.5)  # uninformative -> AUROC == 0.5

    expected_auc1 = roc_auc_score(labels, preds_1)
    diff, p = fast_delong(labels, preds_1, preds_2)

    assert diff == pytest.approx(expected_auc1 - 0.5, abs=1e-9)
    assert (0.0 <= p <= 1.0) or np.isnan(p)


def test_fast_delong_identical_predictors_gives_zero_diff():
    rng = np.random.RandomState(1)
    n = 150
    labels = (rng.rand(n) < 0.5).astype(float)
    preds = rng.rand(n)

    diff, p = fast_delong(labels, preds, preds.copy())

    assert diff == pytest.approx(0.0, abs=1e-9)

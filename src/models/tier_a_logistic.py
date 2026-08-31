"""Extracted unchanged from notebook Sec. 5's Tier A training block.
Shares the StandardScaler fit on TRAIN_FEATURES only, from
src/features/hrv_features.py -- never refit on test data.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.hrv_features import HRV_COLS, fit_hrv_scaler, transform_hrv_features
from src.utils.config import Config


def train_tier_a(train_features: pd.DataFrame, cfg: Config) -> LogisticRegression:
    scaler, train_medians = fit_hrv_scaler(train_features)
    X_train = transform_hrv_features(train_features, scaler, train_medians)
    y_train = train_features["label"].values

    tier_a_model = LogisticRegression(max_iter=1000, C=cfg.lr_tier_a).fit(X_train, y_train)

    print(f"[Tier A] LogisticRegression fit on {X_train.shape[0]:,} segments, {X_train.shape[1]} features")

    return tier_a_model


def tier_a_coefficient_table(tier_a_model: LogisticRegression) -> pd.DataFrame:
    # Logistic regression coefficients are directly interpretable -- report them
    # as a sanity check against the Sec. 2 mechanism (SDNN/RMSSD should matter).
    coef_table = pd.DataFrame({"feature": HRV_COLS, "coefficient": tier_a_model.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False)
    return coef_table

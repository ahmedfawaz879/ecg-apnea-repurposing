"""Extracted unchanged from notebook Sec. 5's Tier B training block.
Shares the StandardScaler fit on TRAIN_FEATURES only, from
src/features/hrv_features.py -- never refit on test data.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features.hrv_features import fit_hrv_scaler, transform_hrv_features
from src.utils.config import Config, SEED


def train_tier_b(train_features: pd.DataFrame, cfg: Config) -> HistGradientBoostingClassifier:
    scaler, train_medians = fit_hrv_scaler(train_features)
    X_train = transform_hrv_features(train_features, scaler, train_medians)
    y_train = train_features["label"].values

    tier_b_model = HistGradientBoostingClassifier(random_state=SEED).fit(X_train, y_train)

    print(f"[Tier B] HistGradientBoostingClassifier fit on {X_train.shape[0]:,} segments")

    return tier_b_model

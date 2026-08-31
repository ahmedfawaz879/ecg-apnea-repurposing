"""Extracted from notebook Sec. 3 ("Grouped train/test split" + "Leakage
check" blocks), wrapped into a named, importable function per the
REPO-INGEST prompt (the notebook itself ran this as script-level code
against module globals). Logic is unchanged.

CRITICAL:
Minutes from the same recording must NEVER occur in both train and test.
We therefore split using recording_id as the group.
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_train_test_split(
    df: pd.DataFrame,
    group_col: str = "recording_id",
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    train_idx, test_idx = next(
        gss.split(
            df,
            groups=df[group_col],
        )
    )

    train_df = (
        df
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    test_df = (
        df
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------
    train_groups = set(train_df[group_col])
    test_groups = set(test_df[group_col])

    overlap = train_groups & test_groups

    assert not overlap, (
        "LEAKAGE: recordings appear in both train and test: "
        f"{sorted(overlap)}"
    )

    return train_df, test_df

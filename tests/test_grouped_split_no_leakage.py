import pandas as pd

from src.data.grouped_splits import group_train_test_split


def test_no_recording_id_in_both_train_and_test():
    """The methodological claim this repository insists on (Abstract,
    Sec. 3): a per-recording split must never let the same recording_id
    appear in both train and test."""
    n_recordings = 20
    minutes_per_recording = 10
    df = pd.DataFrame({
        "recording_id": [f"r{i:02d}" for i in range(n_recordings) for _ in range(minutes_per_recording)],
        "minute_start_sample": list(range(minutes_per_recording)) * n_recordings,
    })

    train_df, test_df = group_train_test_split(df, group_col="recording_id", test_size=0.25, seed=0)

    train_ids = set(train_df["recording_id"])
    test_ids = set(test_df["recording_id"])

    assert train_ids.isdisjoint(test_ids), (
        f"recording_id leaked across train/test: {train_ids & test_ids}"
    )
    assert len(train_df) + len(test_df) == len(df)
    assert len(train_ids) + len(test_ids) == n_recordings

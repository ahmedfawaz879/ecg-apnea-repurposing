"""Extracted unchanged from notebook Sec. 3 ("Loading Recordings -- Real
WFDB Records + Synthetic Fallback")."""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import wfdb

from src.utils.config import Config


# ============================================================
# Apnea-ECG record discovery
# ============================================================

def list_annotated_records(mount_dir: str) -> List[str]:
    """
    Returns base paths (no extension) for records that have a .hea
    header and a .apn apnea annotation file.

    The original PhysioNet Apnea-ECG learning set contains:
        a01-a20
        b01-b05
        c01-c10

    Mirrors may contain additional records, so the caller can optionally
    restrict the discovered records to the expected learning-set IDs.
    """
    mount = Path(mount_dir)

    records = []

    for apn in sorted(mount.glob("**/*.apn")):
        base = apn.with_suffix("")
        hea = base.with_suffix(".hea")

        if hea.exists():
            records.append(str(base))

    return records


# ============================================================
# Real Apnea-ECG loader
# ============================================================

def load_recording(record_base: str, cfg: Config) -> pd.DataFrame:
    """
    Reads one WFDB record + its .apn annotations.

    Returns one row per annotated minute containing:
        - recording_id
        - minute_start_sample
        - label
        - raw_segment
        - center_minute_segment

    IMPORTANT:
    A malformed/corrupt/incompatible WFDB record is skipped rather than
    aborting the entire cohort construction.
    """

    recording_id = Path(record_base).name

    # --------------------------------------------------------
    # Load ECG signal
    # --------------------------------------------------------
    try:
        record = wfdb.rdrecord(record_base)

    except Exception as e:
        print(
            f"[APNEA-ECG][SKIP] failed to read signal "
            f"{recording_id}: {type(e).__name__}: {e}"
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Extract first ECG channel
    # --------------------------------------------------------
    try:
        if record.p_signal is None:
            raise ValueError("WFDB returned p_signal=None")

        if record.p_signal.ndim != 2:
            raise ValueError(
                f"unexpected signal shape: {record.p_signal.shape}"
            )

        if record.p_signal.shape[1] < 1:
            raise ValueError("record contains zero signal channels")

        signal = record.p_signal[:, 0].astype(np.float32)

    except Exception as e:
        print(
            f"[APNEA-ECG][SKIP] invalid signal "
            f"{recording_id}: {type(e).__name__}: {e}"
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Load apnea annotations
    # --------------------------------------------------------
    try:
        ann = wfdb.rdann(record_base, extension="apn")

    except Exception as e:
        print(
            f"[APNEA-ECG][SKIP] failed to read .apn annotations "
            f"{recording_id}: {type(e).__name__}: {e}"
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------
    if len(ann.sample) == 0:
        print(
            f"[APNEA-ECG][SKIP] no annotations found for "
            f"{recording_id}"
        )
        return pd.DataFrame()

    if len(ann.sample) != len(ann.symbol):
        print(
            f"[APNEA-ECG][SKIP] annotation length mismatch "
            f"{recording_id}: "
            f"{len(ann.sample)} samples vs {len(ann.symbol)} symbols"
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Context window
    #
    # For context_minutes=3:
    #
    #   previous minute
    #        +
    #   labeled minute
    #        +
    #   next minute
    #
    # --------------------------------------------------------
    if cfg.context_minutes % 2 == 0:
        raise ValueError(
            f"context_minutes must be odd, got {cfg.context_minutes}"
        )

    context_half_minutes = cfg.context_minutes // 2
    context_half_samples = (
        context_half_minutes * cfg.minute_samples
    )

    expected_segment_len = (
        cfg.context_minutes * cfg.minute_samples
    )

    rows = []

    # --------------------------------------------------------
    # One row per annotated minute
    # --------------------------------------------------------
    for sample_start, symbol in zip(ann.sample, ann.symbol):

        sample_start = int(sample_start)

        # Full context:
        #
        # [sample_start - half_context]
        # ...
        # [sample_start + minute + half_context]
        #
        win_start = sample_start - context_half_samples
        win_end = (
            sample_start
            + cfg.minute_samples
            + context_half_samples
        )

        # Skip minutes too close to recording boundaries.
        if win_start < 0 or win_end > len(signal):
            continue

        # Exact labeled minute.
        center_end = sample_start + cfg.minute_samples

        if center_end > len(signal):
            continue

        raw_segment = signal[win_start:win_end]
        center_minute_segment = signal[
            sample_start:center_end
        ]

        # Defensive shape checks.
        if len(raw_segment) != expected_segment_len:
            continue

        if len(center_minute_segment) != cfg.minute_samples:
            continue

        rows.append({
            "recording_id": recording_id,
            "minute_start_sample": sample_start,

            # Apnea annotation:
            # A = apnea
            # everything else = non-apnea
            "label": 1.0 if symbol == "A" else 0.0,

            # Full context window used for HRV/features and Tier C.
            "raw_segment": raw_segment,

            # Exactly the labeled minute, used by Tier C alone.
            "center_minute_segment": center_minute_segment,
        })

    if not rows:
        print(
            f"[APNEA-ECG][SKIP] no valid full-context minutes "
            f"survived boundary checks for {recording_id}"
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df


# ============================================================
# Synthetic Apnea-ECG fallback
# ============================================================

def make_synthetic_apnea_ecg(
    n_recordings: int,
    minutes_per_recording: int,
    cfg: Config,
    seed: int,
) -> pd.DataFrame:
    """
    Synthesizes an RR-interval-driven ECG-like signal where apnea
    minutes have a slower, more variable heart rate.

    This is intended only for offline pipeline testing and is NOT
    a substitute for the real PhysioNet Apnea-ECG cohort.
    """

    rng = np.random.RandomState(seed)

    all_rows = []

    for r in range(n_recordings):

        recording_id = f"synthetic_{r:02d}"

        base_hr = rng.uniform(60, 80)

        labels = (
            rng.rand(minutes_per_recording) < 0.35
        ).astype(float)

        full_signal_minutes = []

        # ----------------------------------------------------
        # Generate each minute
        # ----------------------------------------------------
        for m in range(minutes_per_recording):

            hr = (
                base_hr
                - 15 * labels[m]
                + rng.normal(0, 3)
            )

            hr = max(hr, 30)

            rr_interval = 60.0 / hr

            n_beats = int(60 / rr_interval)

            t = np.linspace(
                0,
                60,
                cfg.minute_samples,
                endpoint=False,
            )

            # Slightly more timing variability during apnea.
            beat_times = (
                np.cumsum(
                    np.full(n_beats, rr_interval)
                )
                + rng.normal(
                    0,
                    0.02 * (1 + labels[m]),
                    n_beats,
                )
            )

            sig = np.zeros(
                cfg.minute_samples,
                dtype=np.float32,
            )

            for bt in beat_times:

                idx = int(
                    bt / 60 * cfg.minute_samples
                )

                if 0 <= idx < cfg.minute_samples - 5:

                    sig[idx:idx + 5] += np.array(
                        [
                            0.2,
                            1.0,
                            -0.3,
                            0.1,
                            0.05,
                        ],
                        dtype=np.float32,
                    )

            sig += rng.normal(
                0,
                0.03,
                cfg.minute_samples,
            ).astype(np.float32)

            full_signal_minutes.append(sig)

        full_signal = np.concatenate(
            full_signal_minutes
        )

        # ----------------------------------------------------
        # Generate labeled context windows
        # ----------------------------------------------------
        context_half_samples = (
            (cfg.context_minutes // 2)
            * cfg.minute_samples
        )

        for m in range(
            cfg.context_minutes // 2,
            minutes_per_recording - cfg.context_minutes // 2,
        ):

            center_start = (
                m * cfg.minute_samples
            )

            win_start = (
                center_start
                - context_half_samples
            )

            win_end = (
                center_start
                + cfg.minute_samples
                + context_half_samples
            )

            center_end = (
                center_start
                + cfg.minute_samples
            )

            raw_segment = full_signal[
                win_start:win_end
            ]

            center_minute_segment = full_signal[
                center_start:center_end
            ]

            if len(raw_segment) != (
                cfg.context_minutes
                * cfg.minute_samples
            ):
                continue

            if len(center_minute_segment) != (
                cfg.minute_samples
            ):
                continue

            all_rows.append({
                "recording_id": recording_id,
                "minute_start_sample": center_start,
                "label": labels[m],
                "raw_segment": raw_segment,
                "center_minute_segment":
                    center_minute_segment,
            })

    return pd.DataFrame(all_rows)

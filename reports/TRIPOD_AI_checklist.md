# TRIPOD+AI Reporting Checklist

| TRIPOD+AI item | Where addressed |
|---|---|
| Source of data, eligibility criteria | Sec. 2-3 (PhysioNet Apnea-ECG, Kaggle mirror access) |
| Outcome definition | Per-minute apnea/normal from expert `.apn` annotations (Sec. 3) |
| Grouping to prevent leakage | Sec. 3, `GroupShuffleSplit` by `recording_id`, asserted disjoint |
| Feature engineering | Sec. 4 (time+frequency HRV, established QRS detector, not hand-rolled) |
| Model development (3 tiers) | Sec. 5-6 |
| Discrimination + uncertainty | Sec. 7.2 recording-clustered bootstrap AUROC CIs |
| Calibration | Sec. 7.3, Sec. 8 reliability plot |
| Comparison between models | Sec. 8, DeLong pairwise tests |
| Physiological plausibility check | Sec. 5, Tier A coefficient inspection against Sec. 2's CVHR mechanism |
| Limitations | Sec. 11 |

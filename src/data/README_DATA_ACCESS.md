# Data Access

No credentialing needed.

`+ Add Input` → **Datasets** → search `Apnea ECG` → attach **"The Apnea
ECG Database v1.0.0"** (`paulopinheiro/the-apnea-ecg-database-v100`) —
confirmed to directly mirror the official PhysioNet `apnea-ecg/1.0.0`
release (real `.hea`/`.dat` WFDB record pairs plus `.apn` per-minute
annotations, not a reformatted/CSV version).

`ecerulm/apneaecg` is a backup mirror if this one is ever unavailable.

Section 1's auto-discovery (`src/utils/config.py::resolve_apnea_ecg_dir`)
finds whichever mirror is attached by file signature (a directory
containing both `.hea` and `.apn` files), not by a hardcoded name or
path — a lesson from earlier notebooks in this series, where Kaggle's
mount convention for the *same* dataset changed mid-project.

If no matching mount is found (e.g. running outside Kaggle without the
dataset attached), the pipeline falls back to synthetic data for
offline/CI runs — see `src/data/apnea_ecg_loader.py::make_synthetic_apnea_ecg`.
This synthetic mode is for pipeline testing only and must never be used
to report a result.

"""Shared Config and environment setup extracted from notebook Sec. 1
("Environment & Config"). torch/wfdb are declared in requirements.txt, so
the notebook's self-installing import fallbacks are not reproduced here.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    # Data
    apnea_ecg_dir: str = "/kaggle/input/datasets/paulopinheiro/the-apnea-ecg-database-v100"  # confirmed live mount path (see Sec. 2)
    fs: int = 100                       # PhysioNet Apnea-ECG sampling rate
    minute_samples: int = 100 * 60      # samples per minute at 100 Hz
    context_minutes: int = 3            # 3-minute window (label minute +/- 1) for stable HRV estimates -- see Sec. 4
    min_beats_required: int = 20        # skip a segment if fewer R-peaks detected than this (edge-of-recording / noisy)
    # Model
    cnn_hidden: int = 32
    # Training
    batch_size: int = 64
    lr_tier_a: float = 1.0              # sklearn LogisticRegression C is inverse-lambda; not a torch LR
    lr_tier_c: float = 1e-3
    epochs_tier_c: int = 10
    n_bootstrap: int = 500
    ci_alpha: float = 0.05
    test_size: float = 0.25             # fraction of RECORDINGS (not minutes) held out
    seed: int = SEED


CFG = Config()


def _find_apnea_ecg_dir(configured_path: str) -> Optional[str]:
    """Multiple independent Kaggle mirrors of this PhysioNet database exist
    (different uploaders, different folder layouts, some flat, some nested
    under kaggle.com's own datasets/ prefix -- a pattern already seen twice
    in this project series). Checks the configured default, then walks
    /kaggle/input for the first directory containing a WFDB header (.hea)
    paired with an apnea annotation (.apn) file -- the signature that
    unambiguously identifies this specific database regardless of mirror."""
    def _has_signature(p: str) -> bool:
        p = Path(p)
        if not p.exists():
            return False
        return any(p.glob("**/*.apn")) and any(p.glob("**/*.hea"))

    if _has_signature(configured_path):
        return configured_path
    if not Path("/kaggle/input").exists():
        return None
    for candidate in Path("/kaggle/input").iterdir():
        if candidate.is_dir() and _has_signature(str(candidate)):
            print(f"[DATA] found Apnea-ECG database at {candidate} (differs from configured "
                  f"default {configured_path} -- using the discovered path).")
            return str(candidate)
    return None


def resolve_apnea_ecg_dir(cfg: Config) -> bool:
    """Runs the Sec. 1 auto-discovery against `cfg` in place and returns
    True if no real Apnea-ECG mount was found (i.e. synthetic-fallback
    mode is required). Kept as an explicit call rather than an import-time
    side effect so importing this module never touches the filesystem."""
    found = _find_apnea_ecg_dir(cfg.apnea_ecg_dir)
    if found:
        cfg.apnea_ecg_dir = found
    use_synthetic_data = found is None
    print(f"[DATA MODE] {'SYNTHETIC (demo/CI run)' if use_synthetic_data else 'REAL, Kaggle-attached Apnea-ECG detected'}")
    print(f"  apnea_ecg_dir in use: {cfg.apnea_ecg_dir}")
    if use_synthetic_data:
        print("=" * 78)
        print("  ⚠️  SYNTHETIC-DATA MODE: all downstream numbers are placeholders")
        print("      On kaggle.com: '+ Add Input' -> Datasets tab -> search 'Apnea ECG' and")
        print("      attach ANY of the known mirrors (e.g. 'The Apnea-ECG database v1.0.0' by")
        print("      paulopinheiro, or 'apneaecg' by ecerulm) -- auto-discovery above finds")
        print("      whichever one you attach by its .hea/.apn file signature, not by name.")
        print("=" * 78)
    return use_synthetic_data

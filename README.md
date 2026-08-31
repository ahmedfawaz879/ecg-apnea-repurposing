# Repurposing a Cardiac-Monitoring Signal for an Unintended Task
### Single-Lead ECG → Sleep Apnea Detection, With Leakage-Aware Validation, on Real PhysioNet Data

**Author:** Ahmed Fawaz &nbsp;|&nbsp; Port Said University &nbsp;|&nbsp; github.com/ahmedfawaz879

---

## Abstract

**The device-repurposing question, in its cleanest form.** A single-lead
ECG monitor is built to do exactly one thing: capture the heart's
electrical rhythm, for arrhythmia detection. It has no respiratory
sensor, no airflow sensor, no oxygen sensor. And yet the same signal it
was already recording — for a completely different clinical reason — can
be used to detect **sleep apnea**, a condition affecting breathing, not
cardiac rhythm directly. This is the device-repurposing literature
review's Section 16 argument (implantable cardiac devices repurposed via
their incidental telemetry for a diagnosis outside their original
design intent) in its most literal, most reproducible form — and the
hardware involved (single-lead ECG) is now sitting on wrists everywhere
(Apple Watch, Kardia, Fitbit), not locked inside an implanted device.

**The physiological mechanism, stated before any model is built.**
Apneic and hypopneic events trigger repeated brief arousals and
autonomic (vagal/sympathetic) swings, producing a measurable
**cyclic variation of heart rate (CVHR)** — a slowing during the apneic
phase, a surge on resumption of breathing — that shows up in heart-rate
variability (HRV) derived purely from R-R intervals. This is *why* an
ECG-only signal carries respiratory information at all; the model below
is learning to detect this mechanism's fingerprint, not performing magic.

**Data — real, no credentialing.** The PhysioNet Apnea-ECG database
(Penzel et al., 2000) — 70 continuous overnight single-lead ECG
recordings with per-minute expert apnea/normal annotations, openly
licensed, mirrored on Kaggle with no PhysioNet account required.

**The methodological point this notebook insists on.** A documented,
common mistake in this exact literature is splitting train/test **by
minute-segment** rather than **by recording** — since adjacent minutes
of the same overnight recording are highly correlated, a random
per-segment split leaks the same patient's physiology across train and
test, inflating reported accuracy. Every split in this notebook is
**grouped by recording**, and this is treated as a first-class
methodological claim, not a footnote.

**The complexity gradient, again.** Three tiers, same "does complexity
help" question as the earlier VLM projects: **(A)** hand-crafted HRV
features → logistic regression, **(B)** the same features → gradient-
boosted trees, **(C)** raw ECG waveform → a 1D-CNN learning its own
representation. All three evaluated with the same DeLong / clustered-
bootstrap / calibration machinery used throughout this portfolio.

## Key Finding

TODO: fill in from `results/tables/apnea_tier_results.csv` after running
the pipeline against the **real** PhysioNet Apnea-ECG data (synthetic
fallback mode must never be used to report a result — see
`src/data/README_DATA_ACCESS.md`).

> Using only a single-lead ECG's incidental heart-rate-variability signal
> — a signal the recording device was built to capture for cardiac rhythm
> monitoring, not respiratory assessment — per-minute sleep apnea
> detection achieved AUROC **`<fill>`** (Tier A, hand-crafted HRV
> features) vs. **`<fill>`** (Tier C, raw-waveform 1D-CNN), a difference
> of **`<fill>`** (DeLong p=**`<fill>`**), on recordings held out at the
> **recording level** (no patient's data present in both train and test).
> Calibration (ECE) was **`<fill>`** for the best-discriminating tier.

## Setup

```
pip install -r requirements.txt
```

See `src/data/README_DATA_ACCESS.md` for how to attach the PhysioNet
Apnea-ECG database (no credentialing required).

## Reports

- [`reports/MECHANISM_WRITEUP.md`](reports/MECHANISM_WRITEUP.md) — the
  physiological argument for why an ECG carries respiratory information.
- [`reports/TRIPOD_AI_checklist.md`](reports/TRIPOD_AI_checklist.md) —
  TRIPOD+AI reporting checklist, generated from
  `src/evaluation/tripod_ai_report.py`.
- [`src/data/README_DATA_ACCESS.md`](src/data/README_DATA_ACCESS.md) —
  data access workflow.

## Repository Layout

```
ecg-apnea-repurposing/
├── src/
│   ├── data/          <- WFDB + synthetic loading, recording-level grouped splits
│   ├── features/       <- HRV feature extraction (time + frequency domain)
│   ├── models/          <- Tier A (logistic regression), B (gradient-boosted trees), C (1D-CNN)
│   ├── evaluation/      <- DeLong, clustered bootstrap, calibration, TRIPOD+AI report
│   └── utils/           <- Config
├── results/
│   ├── figures/
│   └── tables/
├── reports/
└── tests/
```

## Limitations, Ethics, and Scope

- **Not a clinical tool.** Nothing here is validated for diagnosing sleep
  apnea in a clinical setting; PhysioNet's Apnea-ECG database is a
  research benchmark, not a diagnostic-grade cohort.
- **R-peak detection uses an established library function
  (`wfdb.processing.xqrs_detect`), not a from-scratch QRS detector** —
  correctly scoped for this project's question (does the repurposed
  *signal* carry the diagnostic information, and does model complexity
  help extract it), not a signal-processing algorithms project.
- **The 3-minute context window is wider than the single labeled
  minute** — a deliberate, literature-consistent choice for stable HRV
  estimates, explicitly not a source of label leakage (only raw signal
  context is widened; the predicted label always corresponds to the
  single center minute).
- **35 recordings (or fewer, depending on the Kaggle mirror) is a small
  number of held-out groups** for the test-set bootstrap — recording-
  clustered CIs on a handful of clusters can be wide or unstable; report
  `n_boot_valid` from `clustered_bootstrap_auroc`, not just the point
  estimate, when judging precision.
- **Some minutes are dropped** when fewer than `min_beats_required`
  R-peaks are detected — typically at recording edges or during
  especially noisy segments. This is a real, reported attrition, not
  silently absorbed into the denominator.
- **Single dataset, single population.** One dataset standing in for
  "the general repurposing question" cannot separate a genuine
  physiological finding from this specific cohort's recording
  equipment, scoring conventions, or population characteristics.

## References

1. Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The Apnea-ECG database. Computers in Cardiology 2000;27:255-258.
2. [Device repurposing literature review, Section 16 — CIED/telemetry repurposing framing referenced throughout this notebook.]
3. Guilleminault C, Connolly S, Winkle R, Melvin K, Tilkian A. Cyclical variation of the heart rate in sleep apnoea syndrome. Lancet. 1984;1(8369):126-131.
4. de Chazal P, Heneghan C, Sheridan E, Reilly R, Nolan P, O'Malley M. Automated processing of the single-lead electrocardiogram for the detection of obstructive sleep apnoea. IEEE Trans Biomed Eng. 2003;50(6):686-696.
5. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves. Biometrics. 1988;44(3):837-45.
6. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ. 2024;385:e078378.
7. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation, and clinical use. Circulation. 1996;93(5):1043-1065.

## License

MIT — see [`LICENSE`](LICENSE).

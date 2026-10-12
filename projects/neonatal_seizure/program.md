# Project: neonatal seizure detection (per-second, neonate-independent)

## Data you receive
- `items`: list of recording ids (one recording per neonate).
- `ctx["load_eeg"](items)` -> list of float32 arrays (19 x samples) in microvolts, referential
  montage, rows in `ctx["channels"]` order (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3,
  T4, T5, T6, Fz, Cz, Pz), sampling rate `ctx["fs"]` = 256 Hz. Arrays are memory-mapped; a
  recording is ~1-5 hours long, so process recordings one at a time and keep memory bounded.
- `y_train`: list of int8 arrays, one per training recording, seizure (1) or not (0) for every
  second. For each test recording you must return an array of exactly
  `ctx["n_seconds"](item)` floats (seizure probability per second).
- Metric: AUROC over all test seconds concatenated across neonates. ~10% of seconds are seizure,
  and many neonates have none.

## Open questions worth exploring
Montage (the clinical neonatal bipolar double-banana from these electrodes), filtering, window
length and overlap, features that capture rhythmic evolving discharges (line length, rhythmicity,
autocorrelation, spectral peak/power, entropy, nonlinear energy), per-channel detection then
max/top-k across channels vs global features, per-neonate normalization (background differs
greatly between neonates), temporal context (features of neighbouring seconds) and
post-processing (smoothing, hysteresis), class imbalance, and classifier choice (GBM, SVM,
small CNNs on CPU). Keep the whole CV within ~25 minutes on 8 CPUs.

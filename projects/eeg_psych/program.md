# Project: resting-state EEG -> psychiatric diagnosis (depression, ADHD, schizophrenia)

## Data you receive
- `items`: list of subject ids "dataset:subject". One prediction per subject.
- `ctx["load_eeg"](items, cond)` -> list of float32 arrays (channels x samples) in microvolts, with
  `cond` in `ctx["conditions"]` ("EC" eyes closed, "EO" eyes open). Rows follow `ctx["channels"]`
  (standard 10-20 names, e.g. Fp1, F3, Cz, O2); sampling rate `ctx["fs"]` Hz. Recordings are
  minimally processed (no artifact rejection, original reference; line noise may be present),
  and lengths differ between subjects/datasets (1-15 min). An entry is None if the subject lacks
  that condition. The final champion is also applied to EXTERNAL datasets (other clinics, other
  amplifiers, EC only): robustness to amplitude scale, reference and recording length matters.
- No demographics are given (groups are age-matched where needed).
- Predictions: one float per subject, higher = more likely the disorder. Metric: AUROC.

## Open questions worth exploring
Preprocessing (band-pass, notch, average re-reference, robust artifact/epoch rejection), which
feature families carry signal for THIS disorder: absolute vs relative band power, theta/beta and
other ratios, individual alpha peak frequency, aperiodic (1/f) exponent, frontal alpha asymmetry,
coherence / wPLI / PLV connectivity, covariance matrices in Riemannian tangent space, EEG
microstates (template maps, coverage, duration, transitions), fractal dimension (Higuchi, Katz),
entropies, Hjorth, eyes-closed vs eyes-open reactivity; epoch-level vs subject-level aggregation;
regularized classifiers for small n. Tune only with inner CV on the training fold.

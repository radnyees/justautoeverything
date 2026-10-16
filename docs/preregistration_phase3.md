# Pre-registration: independent Optuna baseline and TDBRAIN protocol probe

Written and committed BEFORE any of the runs below; the commit time is the registration time. Any
deviation will be listed in a "Deviations" section appended to this file, never edited in place.

Context: the existing Optuna baseline searched spaces written after the loop runs, including feature
families the loop found ("loop-informed"). Two of the loop's four significant improvements over the seed,
and the pooled gain's significance, rest on the two TDBRAIN tasks. This phase adds two checks, nothing
else (no new loop runs, no change to existing results).

## 1. Independent Optuna baseline ("Optuna, seed-derived space")

**Question.** How does the loop compare with a hyperparameter/configuration search that uses no
knowledge of the loop's results?

**Search space -- mechanical rule, same for every project** (`projects/<p>/baselines_indep/`):
1. Every numeric constant in the seed solution becomes tunable over a fixed range around the seed value
   (regularization strengths and learning rates on a log scale over roughly x1/30 .. x30; counts such as
   number of genes or trees over a fixed small set that includes the seed value).
2. Every hard-coded choice in the seed among options the task already exposes in `ctx` (fMRI atlas and
   denoising strategy, EEG conditions) becomes tunable over those options; seed feature groups may be
   switched off (subsets of the seed's own features), and the seed's normalization steps may be toggled.
3. The model is chosen from the seed's own model plus ONE generic list used for every classification
   task: L2 logistic regression, random forest, histogram gradient boosting (drug response, a
   regression task: ridge and random forest regression), each with fixed generic hyperparameter ranges.
4. No feature family, transform or model that is not in the seed or in the generic list.

**Protocol.** Identical to the existing Optuna arm: Optuna TPE, sampler seed 1000 + split, 8 random
start-up trials, 20 trials per task x split (splits 1-3), same harness, Dev CV, 1,500 s per trial;
the best Dev trial replaces the seed only if its Dev score is higher; scored once on Eval
(`runs_optuna_indep/`), including external sets.

**Analysis.** Added as arm `optuna_indep` to core/stats.py with comparisons seed -> optuna_indep and
optuna_indep -> loop: hierarchical bootstrap per task with Holm across the 12 tasks, equal-weight pooled
gain over the 11 AUROC tasks. The loop-informed Optuna arm stays in the paper as a second, stronger
baseline. Primary loop-vs-search comparison = loop vs seed-derived Optuna.

## 2. TDBRAIN protocol probe

**Question.** Can non-biological recording properties alone separate patients from controls in the
TDBRAIN tasks (mdd_tdbrain, adhd_tdbrain)? Controls and patients were recruited and recorded separately,
so technical differences could produce the loop's gains.

**Probe features** (per subject, from the same `ctx['load_eeg']` data the solutions see; EC and EO):
* P1 (technical only): recording length per condition; EO-present flag; line-noise relative power
  (49-51 Hz and 59-61 Hz, relative to 1-45 Hz; median over channels); DC offset (median |channel mean|);
  fraction of flat channels (SD < 0.5 uV); fraction of clipped samples (|x| >= 99.9th percentile run of
  identical values, i.e. samples equal to the channel max or min).
* P2 (P1 + gross amplitude, which can also be biological): log median channel SD; log power 55-100 Hz
  excluding 58-62 Hz (muscle/EMG band); log total power 1-45 Hz.

**Model.** Standardization + L2 logistic regression (C = 1.0), no tuning; fitted on each split's dev pool,
scored once on that split's Eval set (splits 1-3), with a subject bootstrap CI.

**Decision rule (fixed now).**
* P1 is "positive" if its Eval AUROC, averaged over the 3 splits, is >= 0.65 AND the pooled bootstrap 95%
  CI excludes 0.5. If positive: (a) the paper states that TDBRAIN case/control labels are predictable
  from recording properties; (b) the loop champion's and seed's Eval predictions are additionally
  reported adjusted for the P1 probe score (same residualization and likelihood-ratio test as
  CONFOUNDS.md); (c) the TDBRAIN results are reported as potentially protocol-confounded.
* If P1 is not positive: report "no evidence that technical recording properties separate the groups",
  with the AUROC and CI (absence of evidence, limited by sample size).
* P2 is reported descriptively only (amplitude and EMG mix biology and hardware).
* Re-defining the tasks (e.g. depression vs ADHD) is NOT part of this phase; if P1 is positive it will be
  proposed separately.

## Not in this phase
ASZED and other external cohorts, stronger-loop variants, basin hopping, ABIDE II, new splits.

## Notes added after running (2026-10-16; nothing above was changed)
* Protocol and decision rule were followed as registered. Result: P1 POSITIVE for both TDBRAIN tasks
  (mean Eval AUROC 0.805 and 0.813, CIs excluding 0.5); adjusted analyses reported in
  results/robustness/PROBE.md as registered.
* Measurement caveat found when interpreting P1: its line-noise features are power at 49-51 / 59-61 Hz
  RELATIVE to 1-45 Hz power, so they also reflect overall EEG power (higher in the MDD group). An
  exploratory per-feature check (core/probe_features.py) shows that ABSOLUTE mains power is not lower in
  patients (MDD: AUROC 0.60, higher in patients); the strongest MDD differences are relative mains power
  and total 1-45 Hz power; for ADHD, recording length (median 120 vs 121 s) and DC offset.
* Exploratory, not registered: frozen seeds and champions re-scored on harmonized recordings (same
  length, mains notched, DC removed; and additionally amplitude-normalized). Results and caveats in
  results/robustness/PROBE.md.
* Independent (seed-derived) Optuna, analysed as registered: all 36 runs x 20 trials completed, 0 failed
  trials; with the seed's own settings each template reproduces the seed's Dev score exactly (checked on
  one task per project before launch). Pooled Eval gain over the seed +0.017 (p = 0.29); loop minus this
  arm +0.033 (p = 0.095); per task (hierarchical bootstrap, Holm across 12): loop better on abnormal_tuab
  (+0.072) and helsinki (+0.067), no task significantly worse. Numbers in results/STATS.md.

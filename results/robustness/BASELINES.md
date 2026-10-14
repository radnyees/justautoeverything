# Baselines with the same budget (Eval, mean over 3 splits)

All arms start from the same seed solution, use the same harness, Dev CV, per-candidate time
limit and the same 20 Dev evaluations, and are scored once on the same Eval sets:
* **One-shot LLM** = round 1 of the best-of-20 run (one proposal; adopted only if it beats the
  seed on Dev).
* **Best-of-20 LLM** = 20 independent proposals, each seeing only the brief + the seed (no
  history, no scores); best Dev candidate adopted if it beats the seed.
* **Optuna, loop-informed space** = 20 Optuna TPE trials of a hand-written parameterized
  pipeline per project (`projects/*/baselines/`); best Dev trial adopted if it beats the seed.
  The optimizer uses no LLM, but the search spaces were written after the loop runs and include
  feature families the loop's champions found (aperiodic slope, coherence, Hjorth parameters,
  per-neonate normalization, ...). It is therefore informed by the loop's findings: a stronger
  baseline than an independent search (conservative for the loop), not an LLM-free one.
* **Loop** = the autoresearch loop (history + scores fed back each round).

| task | seed | One-shot LLM | Best-of-20 LLM (no feedback) | Optuna, loop-informed space | Autoresearch loop |
|---|---|---|---|---|---|
| mdd_tdbrain | 0.850 | 0.864 | 0.884 | 0.946 | **0.954** |
| adhd_tdbrain | 0.785 | 0.785 | 0.785 | 0.918 | **0.922** |
| sz_aseeg | 0.677 | 0.677 | 0.705 | 0.678 | **0.799** |
| abnormal_tuab | 0.847 | 0.866 | 0.888 | 0.889 | **0.918** |
| autism_abide | 0.781 | 0.783 | **0.790** | 0.781 | 0.778 |
| adhd_adhd200 | 0.572 | 0.570 | 0.627 | 0.650 | **0.665** |
| helsinki | 0.827 | 0.827 | 0.827 | 0.824 | **0.899** |
| crc_baxter | 0.795 | **0.789** | 0.787 | 0.759 | 0.740 |
| ibd_pooled | 0.721 | **0.755** | 0.750 | 0.688 | 0.735 |
| plasma_ad | 0.912 | 0.895 | 0.897 | **0.934** | 0.866 |
| csf_ad | 0.913 | 0.913 | 0.891 | **0.971** | 0.954 |
| prism_auc | 0.149 | 0.149 | 0.149 | 0.148 | **0.160** |

One-shot proposals adopted on Dev: 17/36; failed to run: 0/36.

## Paired tests (hierarchical bootstrap over splits, runs and subjects; Holm across 12 tasks)

| comparison | tasks B > A | pooled gain (equal-weight, 11 AUROC tasks) | 95% CI | p | Holm-significant tasks |
|---|---|---|---|---|---|
| seed->loop | 9/12 | +0.050 | +0.006 .. +0.094 | 0.029 | mdd_tdbrain (+0.104), adhd_tdbrain (+0.137), abnormal_tuab (+0.071), helsinki (+0.072) |
| seed->bestof20 | 6/12 | +0.014 | -0.003 .. +0.031 | 0.095 | abnormal_tuab (+0.041) |
| seed->oneshot | 4/12 | +0.004 | -0.005 .. +0.013 | 0.350 | none |
| seed->optuna | 8/12 | +0.033 | -0.004 .. +0.069 | 0.073 | mdd_tdbrain (+0.096), abnormal_tuab (+0.042) |
| bestof20->loop | 8/12 | +0.036 | -0.002 .. +0.075 | 0.063 | adhd_tdbrain (+0.137), helsinki (+0.072) |
| optuna->loop | 8/12 | +0.017 | -0.017 .. +0.051 | 0.281 | abnormal_tuab (+0.029), helsinki (+0.075) |
| oneshot->loop | 8/12 | +0.046 | +0.003 .. +0.089 | 0.037 | mdd_tdbrain (+0.090), adhd_tdbrain (+0.137), abnormal_tuab (+0.052), helsinki (+0.072) |

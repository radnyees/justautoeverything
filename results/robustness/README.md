# Robustness analyses — overview

Index of the five robustness analyses with their headline numbers. All numbers are on held-out
Eval sets (scored once) unless marked Dev. Details, per-task tables and methods are in the linked
files; figures are in `results/figures/`. Regenerate with
`./core/run.sh python core/stats.py && python core/report_robustness.py`.

| # | Analysis | Files | Figure |
|---|---|---|---|
| 1 | Same-budget baselines: one-shot LLM, best-of-20 LLM without feedback, Optuna TPE (no LLM) | [BASELINES.md](BASELINES.md), `baselines.csv` | fig7_baselines |
| 2 | Corrected statistics: Holm/BH across 12 tasks, pooled gain | [../STATS.md](../STATS.md), `../stats_tasks.csv`, `../stats_pooled.json` | fig8_meta_forest |
| 3 | Run-to-run variance (3 independent loops on split 1, 7 tasks) | [VARIANCE.md](VARIANCE.md), `variance.csv` | fig9_variance |
| 4a | Shortcut exploitation under protocol v1; shortcut-only models | [SHORTCUTS.md](SHORTCUTS.md), `shortcuts_raw.json` | fig10_shortcuts |
| 4b | External transfer | [TRANSFER.md](TRANSFER.md), `transfer.csv` | fig11_transfer |
| 5 | Transduction audit of seeds and champions | [AUDIT.md](AUDIT.md), `audit.csv` | fig12_audit |

## Headline numbers

**Pooled Eval gain over the seed** (11 AUROC tasks, equal-weight mean, 95% t-interval across
tasks, two-sided p):

| arm (20 Dev evaluations each) | pooled gain | 95% CI | p |
|---|---|---|---|
| Autoresearch loop | +0.052 | +0.009 .. +0.095 | 0.023 |
| Optuna TPE (no LLM) | +0.035 | +0.000 .. +0.070 | 0.047 |
| Best-of-20 LLM, no feedback | +0.014 | -0.003 .. +0.031 | 0.090 |
| One-shot LLM | +0.002 | -0.005 .. +0.010 | 0.457 |

Head-to-head (loop minus baseline): vs one-shot +0.050 (p = 0.022); vs best-of-20 +0.038
(p = 0.051); vs Optuna +0.017 (95% CI -0.017 .. +0.051, p = 0.29).

**Per task, Holm-adjusted across 12 tasks (loop vs seed):** significant gains on mdd_tdbrain
(+0.104), adhd_tdbrain (+0.137), abnormal_tuab (+0.071), helsinki (+0.072), adhd_adhd200 (+0.093);
a significant *loss* on crc_baxter (-0.055); no significant change on the other 6 tasks.
Loop vs Optuna is Holm-significant only on abnormal_tuab (+0.029) and helsinki (+0.075).
On adhd_tdbrain and helsinki none of the 20 independent (best-of-20) proposals beat the seed on
Dev, while the loop improved both.

**Run-to-run variance:** on split 1, the SD of the Eval gain across 3 independent loops was
0.011-0.044 (median 0.016), smaller than the split-to-split SD (0.002-0.084, median 0.042) for 6 of
7 tasks. The gain had the same sign in all 3 repeats on 3 of 7 tasks (ibd_pooled, mdd_tdbrain,
sz_aseeg: all positive); plasma_ad was never positive (-, 0, -); crc_baxter, csf_ad and prism_auc
flipped sign between repeats.

**Shortcuts (v1):** with no biological input, site + age + sex reach a pooled Eval AUROC of 0.77 on
ADHD-200 (v1 champion: 0.813); within site, age + sex alone still give 0.65. Study identity alone
gives 0.69 pooled AUROC on pooled IBD. Rescored within site, the v1 ADHD-200 champion drops from
0.813 to 0.731 (v2 champion, same split: 0.717).

**External transfer (loop - seed, mean over splits):** mdd_tdbrain -> Mumtaz: in-distribution
+0.104, external AUROC 0.269 -> 0.494 (seed inverted, champion at chance); sz_aseeg -> MSU:
+0.122 in-distribution, 0.696 -> 0.538 externally; sz_aseeg -> RepOD 0.498 -> 0.553;
abnormal_tuab -> official TUAB eval (same hospital): +0.071 vs +0.061 (transfers); crc_baxter
loses on Eval (-0.055) and on all 4 external CRC cohorts.

**Transduction audit:** all 36 seeds are exactly inductive. 9/36 champions have batch-dependent
`featurize` and 11/36 output batch-relative ranks. With a dev-pool reference batch, champion Eval
changes by +0.001 on average (worst: csf_ad split 1, -0.024), i.e. reported Eval scores are not
inflated by Eval-set statistics; without a reference batch (Eval items scored in isolated chunks)
rank-output champions lose up to 0.29, so those pipelines need a reference cohort for
single-patient use.

## Budget used (Modal)

Main study $60.0; robustness analyses $57.7 (best-of-20 and repeat loops, Optuna search, audit,
Eval scoring); total $117.65 (Modal billing, app `jae`). LLM calls (~1,000 for best-of-20 and repeats) used the Claude plan, not Modal.

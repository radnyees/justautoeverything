# Do age and sex explain the Eval results?

Tasks with per-subject age and sex: ADHD-200, ABIDE, TDBRAIN MDD, TDBRAIN ADHD (ASEEG and the
TUAB build have no demographics, so they are not checked). Neither age nor sex is ever given
to a solution, but brain signals encode both, so a model can still learn them. Method in
`core/confounds.py`; within-site metrics for the multi-site tasks.

## Cohort balance

| task | group | n | age, mean (sd) | % male |
|---|---|---|---|---|
| adhd_adhd200 | control | 540 | 12.2 (3.5) | 51 |
| adhd_adhd200 | case | 325 | 11.6 (3.0) | 78 |
| autism_abide | control | 530 | 16.8 (7.5) | 82 |
| autism_abide | case | 505 | 17.1 (8.5) | 88 |
| mdd_tdbrain | control | 128 | 42.3 (12.8) | 43 |
| mdd_tdbrain | case | 151 | 45.2 (11.4) | 50 |
| adhd_tdbrain | control | 119 | 31.9 (9.5) | 43 |
| adhd_tdbrain | case | 51 | 33.1 (9.2) | 57 |

## Eval, mean over 3 splits

| task | arm | AUROC (reported) | age+sex only (fit on dev) | AUROC adjusted for age+sex | rho(prediction, age) in controls | AUROC(prediction -> male) in controls | LR test p (splits) |
|---|---|---|---|---|---|---|---|
| adhd_adhd200 | seed | 0.572 | 0.645 | 0.545 | -0.08 | 0.591 | 0.95 / 0.114 / 0.18 |
| adhd_adhd200 | champion | 0.665 | 0.645 | 0.621 | -0.09 | 0.612 | 0.00043 / 0.237 / 0.117 |
| autism_abide | seed | 0.781 | 0.487 | 0.777 | -0.15 | 0.656 | 1.45e-14 / 6.67e-19 / 6.96e-16 |
| autism_abide | champion | 0.778 | 0.487 | 0.774 | -0.14 | 0.664 | 1.17e-12 / 1.85e-21 / 8.22e-17 |
| mdd_tdbrain | seed | 0.850 | 0.454 | 0.871 | -0.24 | 0.465 | 1.79e-10 / 1.6e-07 / 1.1e-11 |
| mdd_tdbrain | champion | 0.954 | 0.454 | 0.947 | -0.04 | 0.386 | 2.31e-17 / 4.59e-13 / 2.39e-14 |
| adhd_tdbrain | seed | 0.785 | 0.505 | 0.769 | -0.22 | 0.417 | 0.000277 / 0.000245 / 0.0728 |
| adhd_tdbrain | champion | 0.922 | 0.505 | 0.909 | -0.14 | 0.422 | 8.55e-05 / 9e-13 / 9.23e-05 |

## Loop gain before and after adjustment (champion - seed, mean over splits)

| task | gain (complete cases) | gain after age+sex adjustment |
|---|---|---|
| adhd_adhd200 | +0.093 | +0.077 |
| autism_abide | -0.002 | -0.003 |
| mdd_tdbrain | +0.104 | +0.076 |
| adhd_tdbrain | +0.137 | +0.140 |

Reading guide: 'age+sex only' is what demographics alone achieve on the same Eval set;
a prediction that tracks age or sex in controls (|rho| or |AUROC - 0.5| well above 0) carries
demographic signal; 'adjusted' and the LR test ask whether the prediction still separates
cases from controls once age and sex (and site) are accounted for.

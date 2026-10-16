# TDBRAIN protocol probe (pre-registered: docs/preregistration_phase3.md, section 2)

Can recording properties alone (not brain signals) separate patients from controls? Probe fitted
on each split's dev pool, scored once on Eval; mean AUROC over the 3 splits, 95% CI from a bootstrap
over subjects of the union of the Eval sets (2,000 draws).
Decision rule (fixed in advance): P1 positive iff mean AUROC >= 0.65 and the CI excludes 0.5.

| task | probe | Eval AUROC per split | mean | 95% CI | pre-registered verdict |
|---|---|---|---|---|---|
| mdd_tdbrain | P1 | 0.817 / 0.748 / 0.850 | 0.805 | 0.728 .. 0.871 | POSITIVE (protocol signal) |
| mdd_tdbrain | P2 | 0.952 / 0.923 / 0.951 | 0.942 | 0.899 .. 0.975 | descriptive only |
| adhd_tdbrain | P1 | 0.808 / 0.813 / 0.818 | 0.813 | 0.700 .. 0.906 | POSITIVE (protocol signal) |
| adhd_tdbrain | P2 | 0.967 / 0.890 / 0.954 | 0.937 | 0.887 .. 0.978 | descriptive only |

## mdd_tdbrain: P1 positive -> predictions adjusted for the P1 probe score (pre-registered step)

| split | arm | AUROC | AUROC adjusted for probe | LR test p (prediction beyond probe) |
|---|---|---|---|---|
| 1 | seed | 0.871 | 0.831 | 2.03e-07 |
| 1 | champion | 0.984 | 0.942 | 5.09e-15 |
| 2 | seed | 0.792 | 0.768 | 6.28e-06 |
| 2 | champion | 0.933 | 0.868 | 7.64e-10 |
| 3 | seed | 0.887 | 0.810 | 3.61e-08 |
| 3 | champion | 0.944 | 0.891 | 9.18e-11 |

Pre-registered consequence: mdd_tdbrain results are reported as potentially protocol-confounded.

## adhd_tdbrain: P1 positive -> predictions adjusted for the P1 probe score (pre-registered step)

| split | arm | AUROC | AUROC adjusted for probe | LR test p (prediction beyond probe) |
|---|---|---|---|---|
| 1 | seed | 0.838 | 0.718 | 0.0013 |
| 1 | champion | 0.869 | 0.800 | 0.000167 |
| 2 | seed | 0.833 | 0.805 | 8.07e-05 |
| 2 | champion | 1.000 | 0.974 | 1.9e-11 |
| 3 | seed | 0.685 | 0.469 | 0.624 |
| 3 | champion | 0.897 | 0.721 | 0.00246 |

Pre-registered consequence: adhd_tdbrain results are reported as potentially protocol-confounded.


## Exploratory follow-up (not pre-registered): harmonized recordings

Every recording cut to its first 100 s, mains noise notched (50/60 Hz and harmonics), per-channel
DC offset removed, before the frozen solutions see it (probes/harmonize_wrapper.py; solution code
unchanged). The P1 probe re-run on the harmonized data shows how much technical signal remains.

| task | model | Eval AUROC raw (mean of 3 splits) | harmonized | 95% CI harmonized |
|---|---|---|---|---|
| mdd_tdbrain | probeP1 | 0.805 | 0.784 | 0.706 .. 0.853 |
| mdd_tdbrain | seed | 0.850 | 0.878 | 0.823 .. 0.930 |
| mdd_tdbrain | champion | 0.954 | 0.911 | 0.864 .. 0.951 |
| mdd_tdbrain | loop gain (champion - seed) | +0.104 | +0.033 | |
| adhd_tdbrain | probeP1 | 0.813 | 0.658 | 0.547 .. 0.762 |
| adhd_tdbrain | seed | 0.785 | 0.769 | 0.674 .. 0.852 |
| adhd_tdbrain | champion | 0.922 | 0.771 | 0.678 .. 0.851 |
| adhd_tdbrain | loop gain (champion - seed) | +0.137 | +0.002 | |

## Exploratory follow-up (not pre-registered): harmonized AND amplitude-normalized recordings (each scaled to unit median SD)

Every recording cut to its first 100 s, mains noise notched (50/60 Hz and harmonics), per-channel
DC offset removed, then divided by its median channel SD, before the frozen solutions see it (probes/harmonize_wrapper.py; solution code unchanged). The P1
probe re-run on the same data shows how much technical signal remains.

| task | model | Eval AUROC raw (mean of 3 splits) | harmonized | 95% CI harmonized |
|---|---|---|---|---|
| mdd_tdbrain | probeP1 | 0.805 | 0.689 | 0.595 .. 0.777 |
| mdd_tdbrain | seed | 0.850 | 0.878 | 0.823 .. 0.930 |
| mdd_tdbrain | champion | 0.954 | 0.911 | 0.864 .. 0.951 |
| mdd_tdbrain | loop gain (champion - seed) | +0.104 | +0.033 | |
| adhd_tdbrain | probeP1 | 0.813 | 0.608 | 0.493 .. 0.722 |
| adhd_tdbrain | seed | 0.785 | 0.769 | 0.674 .. 0.852 |
| adhd_tdbrain | champion | 0.922 | 0.771 | 0.678 .. 0.851 |
| adhd_tdbrain | loop gain (champion - seed) | +0.137 | +0.002 | |

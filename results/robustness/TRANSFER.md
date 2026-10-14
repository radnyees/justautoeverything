# External transfer -- a finding: the loop's in-distribution gains do not carry over

Seed and loop champion were fitted on the dev pool and applied ONCE, unchanged, to independent
datasets. Shown are ABSOLUTE AUROCs with 95% subject-bootstrap CIs (2,000 draws) against chance
(0.5) -- not 'gains', because a gain over a seed that is worse than chance (Mumtaz) only means
moving toward chance. The TUAB official eval set comes from the same hospital and protocol as
the training data, so it is a within-hospital check, not transfer. No harmonization across
datasets was attempted; acquisition differences (amplifier, reference, montage, recording
length, inferred ASEEG channel order, age range) are an alternative explanation to biology.

Per task (computed from the table below):

* crc_baxter -> crc_zeller (different cohort and sequencing study): loop champion above chance in 3/3 splits; better than the seed in 0/3; mean external AUROC seed 0.826 -> loop 0.799 (in-distribution Eval of the loop: 0.740).
* crc_baxter -> crc_zackular (different cohort and sequencing study): loop champion above chance in 3/3 splits; better than the seed in 0/3; mean external AUROC seed 0.797 -> loop 0.773 (in-distribution Eval of the loop: 0.740).
* crc_baxter -> crc_xiang (different cohort and sequencing study): loop champion above chance in 2/3 splits; better than the seed in 0/3; mean external AUROC seed 0.750 -> loop 0.697 (in-distribution Eval of the loop: 0.740).
* crc_baxter -> crc_zhao (different cohort and sequencing study): loop champion above chance in 3/3 splits; better than the seed in 2/3; mean external AUROC seed 0.703 -> loop 0.694 (in-distribution Eval of the loop: 0.740).
* mdd_tdbrain -> mumtaz (different site and amplifier (adult MDD vs healthy)): loop champion above chance in 0/3 splits; better than the seed in 3/3; mean external AUROC seed 0.269 -> loop 0.494 (in-distribution Eval of the loop: 0.954).
* sz_aseeg -> msu (different site; adolescents; 1-min, 16-channel recordings): loop champion above chance in 1/3 splits; better than the seed in 0/3; mean external AUROC seed 0.696 -> loop 0.538 (in-distribution Eval of the loop: 0.799).
* sz_aseeg -> repod (different site (Warsaw); 14 + 14 adults): loop champion above chance in 0/3 splits; better than the seed in 2/3; mean external AUROC seed 0.498 -> loop 0.553 (in-distribution Eval of the loop: 0.799).
* abnormal_tuab -> tuab_official_eval (same hospital (TUH) and protocol -- a within-hospital check, not transfer): loop champion above chance in 3/3 splits; better than the seed in 3/3; mean external AUROC seed 0.867 -> loop 0.928 (in-distribution Eval of the loop: 0.918).

| task -> external | what differs | split | in-distribution Eval: seed -> loop | external AUROC seed [95% CI] | external AUROC loop [95% CI] | loop vs chance | best-of-20 | Optuna |
|---|---|---|---|---|---|---|---|---|
| crc_baxter -> crc_zeller | different cohort and sequencing study | 1 | 0.854 -> 0.816 | 0.813 [0.72, 0.90] | 0.801 [0.71, 0.89] | above chance | 0.797 | 0.813 |
| crc_baxter -> crc_zackular | different cohort and sequencing study | 1 | 0.854 -> 0.816 | 0.792 [0.67, 0.90] | 0.789 [0.66, 0.90] | above chance | 0.737 | 0.792 |
| crc_baxter -> crc_xiang | different cohort and sequencing study | 1 | 0.854 -> 0.816 | 0.748 [0.58, 0.89] | 0.693 [0.51, 0.85] | above chance | 0.679 | 0.748 |
| crc_baxter -> crc_zhao | different cohort and sequencing study | 1 | 0.854 -> 0.816 | 0.664 [0.56, 0.76] | 0.666 [0.55, 0.76] | above chance | 0.680 | 0.664 |
| crc_baxter -> crc_zeller | different cohort and sequencing study | 2 | 0.756 -> 0.724 | 0.818 [0.73, 0.90] | 0.805 [0.71, 0.90] | above chance | 0.818 | 0.830 |
| crc_baxter -> crc_zackular | different cohort and sequencing study | 2 | 0.756 -> 0.724 | 0.802 [0.68, 0.91] | 0.790 [0.66, 0.90] | above chance | 0.802 | 0.802 |
| crc_baxter -> crc_xiang | different cohort and sequencing study | 2 | 0.756 -> 0.724 | 0.763 [0.59, 0.91] | 0.712 [0.54, 0.87] | above chance | 0.763 | 0.758 |
| crc_baxter -> crc_zhao | different cohort and sequencing study | 2 | 0.756 -> 0.724 | 0.742 [0.64, 0.84] | 0.691 [0.58, 0.79] | above chance | 0.742 | 0.743 |
| crc_baxter -> crc_zeller | different cohort and sequencing study | 3 | 0.774 -> 0.679 | 0.846 [0.76, 0.92] | 0.791 [0.69, 0.88] | above chance | 0.826 | 0.512 |
| crc_baxter -> crc_zackular | different cohort and sequencing study | 3 | 0.774 -> 0.679 | 0.796 [0.67, 0.90] | 0.739 [0.60, 0.86] | above chance | 0.777 | 0.567 |
| crc_baxter -> crc_xiang | different cohort and sequencing study | 3 | 0.774 -> 0.679 | 0.738 [0.56, 0.89] | 0.685 [0.50, 0.84] | not distinguishable from chance | 0.705 | 0.654 |
| crc_baxter -> crc_zhao | different cohort and sequencing study | 3 | 0.774 -> 0.679 | 0.702 [0.59, 0.80] | 0.726 [0.62, 0.82] | above chance | 0.676 | 0.464 |
| mdd_tdbrain -> mumtaz | different site and amplifier (adult MDD vs healthy) | 1 | 0.871 -> 0.984 | 0.250 [0.13, 0.38] | 0.280 [0.15, 0.42] | below chance | 0.283 | 0.719 |
| mdd_tdbrain -> mumtaz | different site and amplifier (adult MDD vs healthy) | 2 | 0.792 -> 0.933 | 0.268 [0.14, 0.40] | 0.587 [0.43, 0.74] | not distinguishable from chance | 0.474 | 0.406 |
| mdd_tdbrain -> mumtaz | different site and amplifier (adult MDD vs healthy) | 3 | 0.887 -> 0.944 | 0.290 [0.15, 0.43] | 0.615 [0.47, 0.75] | not distinguishable from chance | 0.315 | 0.355 |
| sz_aseeg -> msu | different site; adolescents; 1-min, 16-channel recordings | 1 | 0.699 -> 0.744 | 0.696 [0.58, 0.80] | 0.680 [0.56, 0.79] | above chance | 0.625 | 0.659 |
| sz_aseeg -> repod | different site (Warsaw); 14 + 14 adults | 1 | 0.699 -> 0.744 | 0.490 [0.26, 0.71] | 0.531 [0.30, 0.76] | not distinguishable from chance | 0.526 | 0.439 |
| sz_aseeg -> msu | different site; adolescents; 1-min, 16-channel recordings | 2 | 0.583 -> 0.795 | 0.681 [0.56, 0.79] | 0.520 [0.40, 0.65] | not distinguishable from chance | 0.684 | 0.700 |
| sz_aseeg -> repod | different site (Warsaw); 14 + 14 adults | 2 | 0.583 -> 0.795 | 0.500 [0.26, 0.73] | 0.429 [0.20, 0.67] | not distinguishable from chance | 0.510 | 0.548 |
| sz_aseeg -> msu | different site; adolescents; 1-min, 16-channel recordings | 3 | 0.750 -> 0.859 | 0.710 [0.59, 0.81] | 0.414 [0.29, 0.54] | not distinguishable from chance | 0.681 | 0.705 |
| sz_aseeg -> repod | different site (Warsaw); 14 + 14 adults | 3 | 0.750 -> 0.859 | 0.505 [0.26, 0.73] | 0.699 [0.47, 0.89] | not distinguishable from chance | 0.469 | 0.510 |
| abnormal_tuab -> tuab_official_eval | same hospital (TUH) and protocol -- a within-hospital check, not transfer | 1 | 0.840 -> 0.931 | 0.867 [0.82, 0.91] | 0.930 [0.90, 0.96] | above chance | 0.891 | 0.900 |
| abnormal_tuab -> tuab_official_eval | same hospital (TUH) and protocol -- a within-hospital check, not transfer | 2 | 0.829 -> 0.899 | 0.865 [0.82, 0.90] | 0.933 [0.90, 0.96] | above chance | 0.892 | 0.899 |
| abnormal_tuab -> tuab_official_eval | same hospital (TUH) and protocol -- a within-hospital check, not transfer | 3 | 0.872 -> 0.926 | 0.869 [0.83, 0.91] | 0.921 [0.89, 0.95] | above chance | 0.918 | 0.897 |

## Summary (mean over 3 splits)

| task -> external | in-distribution Eval (loop) | external AUROC seed -> loop | change vs in-distribution | splits with loop above chance |
|---|---|---|---|---|
| crc_baxter -> crc_zeller | 0.740 | 0.826 -> 0.799 | +0.059 | 3/3 |
| crc_baxter -> crc_zackular | 0.740 | 0.797 -> 0.773 | +0.033 | 3/3 |
| crc_baxter -> crc_xiang | 0.740 | 0.750 -> 0.697 | -0.043 | 2/3 |
| crc_baxter -> crc_zhao | 0.740 | 0.703 -> 0.694 | -0.045 | 3/3 |
| mdd_tdbrain -> mumtaz | 0.954 | 0.269 -> 0.494 | -0.460 | 0/3 |
| sz_aseeg -> msu | 0.799 | 0.696 -> 0.538 | -0.261 | 1/3 |
| sz_aseeg -> repod | 0.799 | 0.498 -> 0.553 | -0.246 | 0/3 |
| abnormal_tuab -> tuab_official_eval | 0.918 | 0.867 -> 0.928 | +0.009 | 3/3 |

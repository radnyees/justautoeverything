# Corrected statistics (Eval, 12 tasks)

Task gain = mean over the 3 split seeds of (B - A) on each split's held-out Eval set (absolute
scores of A and B per split are listed). Three p-values per task, all two-sided:

* **p (subjects)** -- paired bootstrap resampling SUBJECTS from the union of the 3 overlapping Eval
  sets. It is CONDITIONAL on the fitted models: it asks whether these particular champions beat
  these seeds on new subjects from the same population, and ignores how much the result changes
  with a different split or a different loop run.
* **p (hierarchical)** -- bootstrap that resamples the 3 splits, then (where independent repeat
  loops exist, split 1 of 10 tasks) the loop run, then subjects within the drawn split. It includes
  split-to-split and run-to-run variation; with only 3 splits the split level is coarse.
* **p (split t)** -- one-sample t-test of the 3 split gains (n = 3, low power).

p-values use the (k+1)/(n+1) convention (2,000 draws), so the smallest two-sided p is ~0.001 and the
smallest Holm-adjusted p over 12 tasks is ~0.012. Holm (family-wise) is applied within each
comparison to both bootstrap p-values. Per-task claims should use the hierarchical p; tasks marked
^ are significant only conditionally on the fitted models. Pooled (11 AUROC tasks; prism_auc is
mean per-drug Pearson r): equal-weight mean of task gains, t-interval across tasks; DerSimonian-
Laird random effects (task SE floored at 0.005) as secondary. Baseline arms use the loop's
keep-if-better-on-Dev rule, so every arm is a pipeline chosen without Eval data.

## seed->loop

| task | gain | seed (per split) | loop (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter^ | -0.055 | 0.854 0.756 0.774 | 0.816 0.724 0.679 | -0.108 .. +0.001 | 0.0600 | 0.4798 | 0.0050 | 0.0400 | 0.1111 |
| ibd_pooled [1] | +0.014 | 0.703 0.684 0.777 | 0.662 0.778 0.766 | -0.056 .. +0.102 | 0.7786 | 1.0000 | 0.6197 | 1.0000 | 0.7712 |
| plasma_ad | -0.046 | 0.940 0.889 0.906 | 0.923 0.795 0.880 | -0.117 .. +0.013 | 0.1549 | 0.9295 | 0.0780 | 0.3118 | 0.2020 |
| csf_ad [2] | +0.041 | 0.952 0.913 0.874 | 0.927 0.952 0.984 | -0.037 .. +0.139 | 0.2330 | 0.9345 | 0.2590 | 0.7770 | 0.4038 |
| prism_auc | +0.012 | 0.168 0.156 0.122 | 0.182 0.168 0.131 | -0.006 .. +0.020 | 0.2079 | 0.9345 | 0.0270 | 0.1619 | 0.0130 |
| autism_abide [3] | -0.002 | 0.775 0.790 0.778 | 0.747 0.817 0.772 | -0.046 .. +0.043 | 0.9125 | 1.0000 | 0.9145 | 1.0000 | 0.9003 |
| adhd_adhd200^ [4] | +0.093 | 0.453 0.628 0.635 | 0.717 0.629 0.649 | -0.022 .. +0.219 | 0.1869 | 0.9345 | 0.0050 | 0.0400 | 0.3894 |
| mdd_tdbrain | +0.104 | 0.871 0.792 0.887 | 0.984 0.933 0.944 | +0.044 .. +0.170 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0530 |
| adhd_tdbrain | +0.137 | 0.838 0.833 0.685 | 0.869 1.000 0.897 | +0.025 .. +0.259 | 0.0040 | 0.0360 | 0.0010 | 0.0120 | 0.1294 |
| sz_aseeg [5] | +0.122 | 0.699 0.583 0.750 | 0.744 0.795 0.859 | -0.008 .. +0.259 | 0.0640 | 0.4798 | 0.0330 | 0.1649 | 0.1288 |
| abnormal_tuab | +0.071 | 0.840 0.829 0.872 | 0.931 0.899 0.926 | +0.051 .. +0.091 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0213 |
| helsinki | +0.072 | 0.852 0.848 0.780 | 0.934 0.896 0.867 | +0.033 .. +0.106 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0276 |

[1] ibd_pooled: split gains change sign (-0.041 +0.093 -0.011).
[2] csf_ad: split gains change sign (-0.026 +0.039 +0.110).
[3] autism_abide: split gains change sign (-0.028 +0.027 -0.005).
[4] adhd_adhd200: seed Eval on split 1 was 0.453 (below 0.6, the lowest of the 3 splits); split gains +0.264 +0.001 +0.015, so the mean gain leans on that split. adhd_adhd200: age + sex alone give a comparable within-site AUROC; see results/robustness/CONFOUNDS.md for the demographic floor per split.
[5] sz_aseeg: seed Eval on split 2 was 0.583 (below 0.6, the lowest of the 3 splits); split gains +0.045 +0.212 +0.109, so the mean gain leans on that split.

Pooled (k=11 AUROC tasks): **+0.050** AUROC, equal-weight mean (95% CI +0.006 .. +0.094, t-test across tasks p = 0.0293). Random effects (DL) +0.044 (95% CI +0.009 .. +0.079, p = 0.0142; I^2 = 85%); 8/11 tasks positive (sign test p = 0.227, Wilcoxon p = 0.042).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.050 | +0.006 .. +0.094 | 0.029 |
| TDBRAIN counted as one dataset | 10 | +0.043 | -0.003 .. +0.089 | 0.062 |
| without TDBRAIN MDD + ADHD | 9 | +0.034 | -0.013 .. +0.082 | 0.130 |
| without ADHD-200 | 10 | +0.046 | -0.002 .. +0.094 | 0.060 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.027 | -0.024 .. +0.078 | 0.250 |
| leave out crc_baxter | 10 | +0.061 | +0.019 .. +0.102 | 0.009 |
| leave out ibd_pooled | 10 | +0.054 | +0.005 .. +0.102 | 0.033 |
| leave out plasma_ad | 10 | +0.060 | +0.017 .. +0.103 | 0.012 |
| leave out csf_ad | 10 | +0.051 | +0.002 .. +0.100 | 0.044 |
| leave out autism_abide | 10 | +0.055 | +0.008 .. +0.103 | 0.027 |
| leave out adhd_adhd200 | 10 | +0.046 | -0.002 .. +0.094 | 0.060 |
| leave out mdd_tdbrain | 10 | +0.045 | -0.003 .. +0.092 | 0.062 |
| leave out adhd_tdbrain | 10 | +0.041 | -0.003 .. +0.086 | 0.063 |
| leave out sz_aseeg | 10 | +0.043 | -0.003 .. +0.089 | 0.064 |
| leave out abnormal_tuab | 10 | +0.048 | -0.001 .. +0.097 | 0.054 |
| leave out helsinki | 10 | +0.048 | -0.001 .. +0.097 | 0.054 |

## seed->bestof20

| task | gain | seed (per split) | bestof20 (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter | -0.007 | 0.854 0.756 0.774 | 0.858 0.756 0.749 | -0.034 .. +0.017 | 0.5917 | 1.0000 | 0.4678 | 1.0000 | 0.5073 |
| ibd_pooled | +0.029 | 0.703 0.684 0.777 | 0.679 0.746 0.826 | -0.032 .. +0.088 | 0.3468 | 1.0000 | 0.2199 | 1.0000 | 0.3907 |
| plasma_ad | -0.014 | 0.940 0.889 0.906 | 0.906 0.880 0.906 | -0.065 .. +0.028 | 0.5327 | 1.0000 | 0.6067 | 1.0000 | 0.2999 |
| csf_ad | -0.022 | 0.952 0.913 0.874 | 0.952 0.847 0.874 | -0.099 .. +0.016 | 0.8356 | 1.0000 | 0.4260 | 1.0000 | 0.4226 |
| prism_auc | -0.000 | 0.168 0.156 0.122 | 0.168 0.156 0.122 | -0.004 .. +0.004 | 1.0000 | 1.0000 | 0.9625 | 1.0000 | 0.4226 |
| autism_abide | +0.009 | 0.775 0.790 0.778 | 0.780 0.812 0.777 | -0.029 .. +0.046 | 0.6307 | 1.0000 | 0.6277 | 1.0000 | 0.3248 |
| adhd_adhd200 | +0.055 | 0.453 0.628 0.635 | 0.627 0.612 0.644 | -0.033 .. +0.149 | 0.3738 | 1.0000 | 0.0730 | 0.8026 | 0.4495 |
| mdd_tdbrain | +0.034 | 0.871 0.792 0.887 | 0.931 0.817 0.905 | -0.031 .. +0.098 | 0.2599 | 1.0000 | 0.3658 | 1.0000 | 0.1204 |
| adhd_tdbrain | +0.000 | 0.838 0.833 0.685 | 0.838 0.833 0.685 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| sz_aseeg | +0.028 | 0.699 0.583 0.750 | 0.724 0.609 0.782 | -0.035 .. +0.089 | 0.3148 | 1.0000 | 0.3658 | 1.0000 | 0.0059 |
| abnormal_tuab | +0.041 | 0.840 0.829 0.872 | 0.883 0.851 0.929 | +0.020 .. +0.061 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0601 |
| helsinki | +0.000 | 0.852 0.848 0.780 | 0.852 0.848 0.780 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |

Pooled (k=11 AUROC tasks): **+0.014** AUROC, equal-weight mean (95% CI -0.003 .. +0.031, t-test across tasks p = 0.0955). Random effects (DL) +0.011 (95% CI -0.004 .. +0.025, p = 0.1399; I^2 = 73%); 6/11 tasks positive (sign test p = 1.000, Wilcoxon p = 0.098).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.014 | -0.003 .. +0.031 | 0.095 |
| TDBRAIN counted as one dataset | 10 | +0.014 | -0.004 .. +0.031 | 0.121 |
| without TDBRAIN MDD + ADHD | 9 | +0.013 | -0.007 .. +0.033 | 0.176 |
| without ADHD-200 | 10 | +0.010 | -0.006 .. +0.025 | 0.196 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.008 | -0.011 .. +0.027 | 0.361 |
| leave out crc_baxter | 10 | +0.016 | -0.002 .. +0.034 | 0.077 |
| leave out ibd_pooled | 10 | +0.012 | -0.006 .. +0.031 | 0.164 |
| leave out plasma_ad | 10 | +0.017 | -0.001 .. +0.034 | 0.059 |
| leave out csf_ad | 10 | +0.017 | +0.001 .. +0.034 | 0.040 |
| leave out autism_abide | 10 | +0.014 | -0.004 .. +0.033 | 0.118 |
| leave out adhd_adhd200 | 10 | +0.010 | -0.006 .. +0.025 | 0.196 |
| leave out mdd_tdbrain | 10 | +0.012 | -0.006 .. +0.030 | 0.174 |
| leave out adhd_tdbrain | 10 | +0.015 | -0.003 .. +0.034 | 0.095 |
| leave out sz_aseeg | 10 | +0.012 | -0.006 .. +0.031 | 0.162 |
| leave out abnormal_tuab | 10 | +0.011 | -0.006 .. +0.029 | 0.184 |
| leave out helsinki | 10 | +0.015 | -0.003 .. +0.034 | 0.095 |

## seed->oneshot

| task | gain | seed (per split) | oneshot (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter | -0.006 | 0.854 0.756 0.774 | 0.854 0.756 0.756 | -0.034 .. +0.012 | 0.9525 | 1.0000 | 0.5907 | 1.0000 | 0.4226 |
| ibd_pooled | +0.033 | 0.703 0.684 0.777 | 0.686 0.757 0.822 | -0.035 .. +0.092 | 0.3078 | 1.0000 | 0.2419 | 1.0000 | 0.3338 |
| plasma_ad | -0.017 | 0.940 0.889 0.906 | 0.906 0.880 0.897 | -0.065 .. +0.014 | 0.3778 | 1.0000 | 0.5167 | 1.0000 | 0.1835 |
| csf_ad | +0.000 | 0.952 0.913 0.874 | 0.952 0.913 0.874 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| prism_auc | +0.000 | 0.168 0.156 0.122 | 0.168 0.156 0.122 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| autism_abide | +0.002 | 0.775 0.790 0.778 | 0.780 0.795 0.774 | -0.028 .. +0.032 | 0.8396 | 1.0000 | 0.8846 | 1.0000 | 0.4872 |
| adhd_adhd200 | -0.002 | 0.453 0.628 0.635 | 0.453 0.620 0.637 | -0.022 .. +0.023 | 0.9995 | 1.0000 | 0.9805 | 1.0000 | 0.5914 |
| mdd_tdbrain | +0.014 | 0.871 0.792 0.887 | 0.871 0.817 0.905 | -0.037 .. +0.075 | 0.6547 | 1.0000 | 0.6247 | 1.0000 | 0.1982 |
| adhd_tdbrain | +0.000 | 0.838 0.833 0.685 | 0.838 0.833 0.685 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| sz_aseeg | +0.000 | 0.699 0.583 0.750 | 0.699 0.583 0.750 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| abnormal_tuab^ | +0.020 | 0.840 0.829 0.872 | 0.874 0.847 0.879 | +0.003 .. +0.036 | 0.0130 | 0.1559 | 0.0020 | 0.0240 | 0.1300 |
| helsinki | +0.000 | 0.852 0.848 0.780 | 0.852 0.848 0.780 | +0.000 .. +0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |

Pooled (k=11 AUROC tasks): **+0.004** AUROC, equal-weight mean (95% CI -0.005 .. +0.013, t-test across tasks p = 0.3500). Random effects (DL) +0.002 (95% CI -0.003 .. +0.007, p = 0.3367; I^2 = 16%); 4/11 tasks positive (sign test p = 0.549, Wilcoxon p = 0.469).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.004 | -0.005 .. +0.013 | 0.350 |
| TDBRAIN counted as one dataset | 10 | +0.004 | -0.006 .. +0.014 | 0.420 |
| without TDBRAIN MDD + ADHD | 9 | +0.003 | -0.008 .. +0.015 | 0.515 |
| without ADHD-200 | 10 | +0.005 | -0.006 .. +0.015 | 0.328 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.004 | -0.009 .. +0.017 | 0.489 |
| leave out crc_baxter | 10 | +0.005 | -0.005 .. +0.015 | 0.284 |
| leave out ibd_pooled | 10 | +0.001 | -0.006 .. +0.008 | 0.739 |
| leave out plasma_ad | 10 | +0.006 | -0.003 .. +0.015 | 0.150 |
| leave out csf_ad | 10 | +0.004 | -0.006 .. +0.015 | 0.353 |
| leave out autism_abide | 10 | +0.004 | -0.006 .. +0.014 | 0.379 |
| leave out adhd_adhd200 | 10 | +0.005 | -0.006 .. +0.015 | 0.328 |
| leave out mdd_tdbrain | 10 | +0.003 | -0.007 .. +0.013 | 0.511 |
| leave out adhd_tdbrain | 10 | +0.004 | -0.006 .. +0.015 | 0.353 |
| leave out sz_aseeg | 10 | +0.004 | -0.006 .. +0.015 | 0.353 |
| leave out abnormal_tuab | 10 | +0.002 | -0.007 .. +0.012 | 0.570 |
| leave out helsinki | 10 | +0.004 | -0.006 .. +0.015 | 0.353 |

## seed->optuna

| task | gain | seed (per split) | optuna (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter | -0.036 | 0.854 0.756 0.774 | 0.854 0.729 0.694 | -0.098 .. +0.001 | 0.1339 | 1.0000 | 0.0710 | 0.5677 | 0.2713 |
| ibd_pooled | -0.034 | 0.703 0.684 0.777 | 0.559 0.713 0.792 | -0.139 .. +0.067 | 0.5537 | 1.0000 | 0.2939 | 1.0000 | 0.6071 |
| plasma_ad | +0.023 | 0.940 0.889 0.906 | 0.923 0.889 0.991 | -0.043 .. +0.115 | 0.5907 | 1.0000 | 0.4648 | 1.0000 | 0.5471 |
| csf_ad | +0.058 | 0.952 0.913 0.874 | 1.000 0.938 0.977 | -0.027 .. +0.146 | 0.1559 | 1.0000 | 0.1460 | 1.0000 | 0.1277 |
| prism_auc | -0.001 | 0.168 0.156 0.122 | 0.168 0.161 0.114 | -0.013 .. +0.010 | 0.8696 | 1.0000 | 0.8586 | 1.0000 | 0.8123 |
| autism_abide | +0.001 | 0.775 0.790 0.778 | 0.772 0.811 0.761 | -0.041 .. +0.042 | 0.9815 | 1.0000 | 0.9655 | 1.0000 | 0.9588 |
| adhd_adhd200 | +0.078 | 0.453 0.628 0.635 | 0.689 0.621 0.641 | -0.032 .. +0.202 | 0.2929 | 1.0000 | 0.0390 | 0.3508 | 0.4266 |
| mdd_tdbrain | +0.096 | 0.871 0.792 0.887 | 0.970 0.919 0.949 | +0.047 .. +0.152 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0365 |
| adhd_tdbrain | +0.132 | 0.838 0.833 0.685 | 0.967 0.913 0.874 | +0.040 .. +0.244 | 0.0060 | 0.0600 | 0.0090 | 0.0900 | 0.0534 |
| sz_aseeg | +0.001 | 0.699 0.583 0.750 | 0.660 0.625 0.750 | -0.086 .. +0.086 | 0.9935 | 1.0000 | 0.9885 | 1.0000 | 0.9674 |
| abnormal_tuab | +0.042 | 0.840 0.829 0.872 | 0.885 0.878 0.904 | +0.027 .. +0.057 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0144 |
| helsinki | -0.003 | 0.852 0.848 0.780 | 0.839 0.820 0.812 | -0.038 .. +0.042 | 0.9765 | 1.0000 | 0.9715 | 1.0000 | 0.8901 |

Pooled (k=11 AUROC tasks): **+0.033** AUROC, equal-weight mean (95% CI -0.004 .. +0.069, t-test across tasks p = 0.0727). Random effects (DL) +0.025 (95% CI -0.001 .. +0.052, p = 0.0619; I^2 = 72%); 8/11 tasks positive (sign test p = 0.227, Wilcoxon p = 0.102).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.033 | -0.004 .. +0.069 | 0.073 |
| TDBRAIN counted as one dataset | 10 | +0.025 | -0.010 .. +0.059 | 0.145 |
| without TDBRAIN MDD + ADHD | 9 | +0.015 | -0.016 .. +0.045 | 0.298 |
| without ADHD-200 | 10 | +0.028 | -0.011 .. +0.067 | 0.138 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.007 | -0.021 .. +0.034 | 0.592 |
| leave out crc_baxter | 10 | +0.040 | +0.002 .. +0.077 | 0.039 |
| leave out ibd_pooled | 10 | +0.039 | +0.002 .. +0.077 | 0.041 |
| leave out plasma_ad | 10 | +0.034 | -0.007 .. +0.074 | 0.094 |
| leave out csf_ad | 10 | +0.030 | -0.010 .. +0.070 | 0.125 |
| leave out autism_abide | 10 | +0.036 | -0.004 .. +0.076 | 0.073 |
| leave out adhd_adhd200 | 10 | +0.028 | -0.011 .. +0.067 | 0.138 |
| leave out mdd_tdbrain | 10 | +0.026 | -0.011 .. +0.064 | 0.147 |
| leave out adhd_tdbrain | 10 | +0.023 | -0.010 .. +0.055 | 0.145 |
| leave out sz_aseeg | 10 | +0.036 | -0.004 .. +0.076 | 0.073 |
| leave out abnormal_tuab | 10 | +0.032 | -0.009 .. +0.072 | 0.111 |
| leave out helsinki | 10 | +0.036 | -0.004 .. +0.076 | 0.069 |

## bestof20->loop

| task | gain | bestof20 (per split) | loop (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter^ | -0.048 | 0.858 0.756 0.749 | 0.816 0.724 0.679 | -0.087 .. -0.000 | 0.0510 | 0.4588 | 0.0050 | 0.0450 | 0.0521 |
| ibd_pooled | -0.015 | 0.679 0.746 0.826 | 0.662 0.778 0.766 | -0.078 .. +0.052 | 0.6937 | 1.0000 | 0.4978 | 0.5717 | 0.6211 |
| plasma_ad | -0.031 | 0.906 0.880 0.906 | 0.923 0.795 0.880 | -0.106 .. +0.024 | 0.3458 | 1.0000 | 0.0910 | 0.3440 | 0.4026 |
| csf_ad | +0.063 | 0.952 0.847 0.874 | 0.927 0.952 0.984 | -0.024 .. +0.170 | 0.1390 | 0.8156 | 0.0860 | 0.3440 | 0.2904 |
| prism_auc | +0.012 | 0.168 0.156 0.122 | 0.182 0.168 0.131 | -0.007 .. +0.021 | 0.2359 | 0.9435 | 0.0520 | 0.2599 | 0.0112 |
| autism_abide | -0.011 | 0.780 0.812 0.777 | 0.747 0.817 0.772 | -0.040 .. +0.013 | 0.4378 | 1.0000 | 0.2859 | 0.5717 | 0.4159 |
| adhd_adhd200 | +0.038 | 0.627 0.612 0.644 | 0.717 0.629 0.649 | -0.010 .. +0.091 | 0.1359 | 0.8156 | 0.0370 | 0.2219 | 0.2916 |
| mdd_tdbrain^ | +0.070 | 0.931 0.817 0.905 | 0.984 0.933 0.944 | +0.011 .. +0.129 | 0.0090 | 0.0900 | 0.0060 | 0.0480 | 0.0976 |
| adhd_tdbrain | +0.137 | 0.838 0.833 0.685 | 0.869 1.000 0.897 | +0.025 .. +0.259 | 0.0040 | 0.0440 | 0.0010 | 0.0120 | 0.1294 |
| sz_aseeg | +0.094 | 0.724 0.609 0.782 | 0.744 0.795 0.859 | -0.005 .. +0.222 | 0.0780 | 0.5757 | 0.0260 | 0.1819 | 0.1943 |
| abnormal_tuab^ | +0.031 | 0.883 0.851 0.929 | 0.931 0.899 0.926 | -0.002 .. +0.055 | 0.0720 | 0.5757 | 0.0010 | 0.0120 | 0.2112 |
| helsinki | +0.072 | 0.852 0.848 0.780 | 0.934 0.896 0.867 | +0.033 .. +0.106 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0276 |

Pooled (k=11 AUROC tasks): **+0.036** AUROC, equal-weight mean (95% CI -0.002 .. +0.075, t-test across tasks p = 0.0635). Random effects (DL) +0.027 (95% CI +0.002 .. +0.053, p = 0.0377; I^2 = 83%); 7/11 tasks positive (sign test p = 0.549, Wilcoxon p = 0.083).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.036 | -0.002 .. +0.075 | 0.063 |
| TDBRAIN counted as one dataset | 10 | +0.030 | -0.009 .. +0.068 | 0.116 |
| without TDBRAIN MDD + ADHD | 9 | +0.021 | -0.017 .. +0.060 | 0.235 |
| without ADHD-200 | 10 | +0.036 | -0.007 .. +0.080 | 0.093 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.019 | -0.025 .. +0.064 | 0.337 |
| leave out crc_baxter | 10 | +0.045 | +0.007 .. +0.083 | 0.026 |
| leave out ibd_pooled | 10 | +0.041 | -0.000 .. +0.083 | 0.050 |
| leave out plasma_ad | 10 | +0.043 | +0.003 .. +0.083 | 0.038 |
| leave out csf_ad | 10 | +0.034 | -0.009 .. +0.076 | 0.111 |
| leave out autism_abide | 10 | +0.041 | -0.001 .. +0.083 | 0.054 |
| leave out adhd_adhd200 | 10 | +0.036 | -0.007 .. +0.080 | 0.093 |
| leave out mdd_tdbrain | 10 | +0.033 | -0.010 .. +0.076 | 0.115 |
| leave out adhd_tdbrain | 10 | +0.026 | -0.009 .. +0.062 | 0.129 |
| leave out sz_aseeg | 10 | +0.030 | -0.010 .. +0.071 | 0.127 |
| leave out abnormal_tuab | 10 | +0.037 | -0.007 .. +0.080 | 0.088 |
| leave out helsinki | 10 | +0.033 | -0.010 .. +0.075 | 0.116 |

## optuna->loop

| task | gain | optuna (per split) | loop (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter | -0.020 | 0.854 0.729 0.694 | 0.816 0.724 0.679 | -0.068 .. +0.037 | 0.5777 | 1.0000 | 0.4578 | 1.0000 | 0.1849 |
| ibd_pooled | +0.047 | 0.559 0.713 0.792 | 0.662 0.778 0.766 | -0.031 .. +0.128 | 0.2669 | 1.0000 | 0.0900 | 0.6297 | 0.3437 |
| plasma_ad | -0.068 | 0.923 0.889 0.991 | 0.923 0.795 0.880 | -0.165 .. +0.015 | 0.1149 | 1.0000 | 0.0220 | 0.2199 | 0.1863 |
| csf_ad | -0.017 | 1.000 0.938 0.977 | 0.927 0.952 0.984 | -0.091 .. +0.071 | 0.8740 | 1.0000 | 0.6090 | 1.0000 | 0.6073 |
| prism_auc | +0.013 | 0.168 0.161 0.114 | 0.182 0.168 0.131 | -0.006 .. +0.025 | 0.2329 | 1.0000 | 0.0340 | 0.3058 | 0.0570 |
| autism_abide | -0.003 | 0.772 0.811 0.761 | 0.747 0.817 0.772 | -0.033 .. +0.025 | 0.8666 | 1.0000 | 0.8166 | 1.0000 | 0.8208 |
| adhd_adhd200 | +0.015 | 0.689 0.621 0.641 | 0.717 0.629 0.649 | -0.023 .. +0.059 | 0.4048 | 1.0000 | 0.4028 | 1.0000 | 0.1459 |
| mdd_tdbrain | +0.008 | 0.970 0.919 0.949 | 0.984 0.933 0.944 | -0.019 .. +0.034 | 0.6817 | 1.0000 | 0.5147 | 1.0000 | 0.3625 |
| adhd_tdbrain | +0.004 | 0.967 0.913 0.874 | 0.869 1.000 0.897 | -0.114 .. +0.102 | 0.9505 | 1.0000 | 0.8626 | 1.0000 | 0.9442 |
| sz_aseeg | +0.121 | 0.660 0.625 0.750 | 0.744 0.795 0.859 | +0.004 .. +0.245 | 0.0440 | 0.4398 | 0.0630 | 0.5037 | 0.0423 |
| abnormal_tuab | +0.029 | 0.885 0.878 0.904 | 0.931 0.899 0.926 | +0.014 .. +0.043 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0705 |
| helsinki | +0.075 | 0.839 0.820 0.812 | 0.934 0.896 0.867 | +0.032 .. +0.103 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0241 |

Pooled (k=11 AUROC tasks): **+0.017** AUROC, equal-weight mean (95% CI -0.017 .. +0.051, t-test across tasks p = 0.2812). Random effects (DL) +0.015 (95% CI -0.004 .. +0.034, p = 0.1157; I^2 = 67%); 7/11 tasks positive (sign test p = 0.549, Wilcoxon p = 0.320).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.017 | -0.017 .. +0.051 | 0.281 |
| TDBRAIN counted as one dataset | 10 | +0.019 | -0.020 .. +0.057 | 0.299 |
| without TDBRAIN MDD + ADHD | 9 | +0.020 | -0.023 .. +0.063 | 0.319 |
| without ADHD-200 | 10 | +0.018 | -0.021 .. +0.056 | 0.323 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.021 | -0.030 .. +0.071 | 0.365 |
| leave out crc_baxter | 10 | +0.021 | -0.016 .. +0.058 | 0.230 |
| leave out ibd_pooled | 10 | +0.014 | -0.023 .. +0.052 | 0.407 |
| leave out plasma_ad | 10 | +0.026 | -0.006 .. +0.058 | 0.096 |
| leave out csf_ad | 10 | +0.021 | -0.016 .. +0.058 | 0.237 |
| leave out autism_abide | 10 | +0.019 | -0.018 .. +0.057 | 0.275 |
| leave out adhd_adhd200 | 10 | +0.018 | -0.021 .. +0.056 | 0.323 |
| leave out mdd_tdbrain | 10 | +0.018 | -0.020 .. +0.056 | 0.304 |
| leave out adhd_tdbrain | 10 | +0.019 | -0.019 .. +0.057 | 0.295 |
| leave out sz_aseeg | 10 | +0.007 | -0.021 .. +0.035 | 0.584 |
| leave out abnormal_tuab | 10 | +0.016 | -0.022 .. +0.054 | 0.361 |
| leave out helsinki | 10 | +0.012 | -0.024 .. +0.047 | 0.476 |

## oneshot->loop

| task | gain | oneshot (per split) | loop (per split) | 95% CI (hierarchical) | p (hierarchical) | Holm | p (subjects) | Holm | p (split t) |
|---|---|---|---|---|---|---|---|---|---|
| crc_baxter^ | -0.049 | 0.854 0.756 0.756 | 0.816 0.724 0.679 | -0.091 .. +0.002 | 0.0640 | 0.5117 | 0.0030 | 0.0210 | 0.0739 |
| ibd_pooled | -0.020 | 0.686 0.757 0.822 | 0.662 0.778 0.766 | -0.084 .. +0.044 | 0.6017 | 1.0000 | 0.4618 | 0.9235 | 0.4676 |
| plasma_ad | -0.028 | 0.906 0.880 0.897 | 0.923 0.795 0.880 | -0.104 .. +0.029 | 0.4388 | 1.0000 | 0.1769 | 0.7076 | 0.4444 |
| csf_ad | +0.041 | 0.952 0.913 0.874 | 0.927 0.952 0.984 | -0.037 .. +0.139 | 0.2330 | 1.0000 | 0.2590 | 0.7770 | 0.4038 |
| prism_auc | +0.012 | 0.168 0.156 0.122 | 0.182 0.168 0.131 | -0.006 .. +0.020 | 0.2079 | 1.0000 | 0.0270 | 0.1619 | 0.0130 |
| autism_abide | -0.005 | 0.780 0.795 0.774 | 0.747 0.817 0.772 | -0.040 .. +0.036 | 0.7676 | 1.0000 | 0.7646 | 0.9235 | 0.7989 |
| adhd_adhd200^ | +0.095 | 0.453 0.620 0.637 | 0.717 0.629 0.649 | -0.019 .. +0.219 | 0.1729 | 1.0000 | 0.0020 | 0.0160 | 0.3756 |
| mdd_tdbrain | +0.090 | 0.871 0.817 0.905 | 0.984 0.933 0.944 | +0.025 .. +0.147 | 0.0040 | 0.0400 | 0.0010 | 0.0120 | 0.0702 |
| adhd_tdbrain | +0.137 | 0.838 0.833 0.685 | 0.869 1.000 0.897 | +0.025 .. +0.259 | 0.0040 | 0.0400 | 0.0010 | 0.0120 | 0.1294 |
| sz_aseeg | +0.122 | 0.699 0.583 0.750 | 0.744 0.795 0.859 | -0.008 .. +0.259 | 0.0640 | 0.5117 | 0.0330 | 0.1649 | 0.1288 |
| abnormal_tuab | +0.052 | 0.874 0.847 0.879 | 0.931 0.899 0.926 | +0.038 .. +0.063 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0028 |
| helsinki | +0.072 | 0.852 0.848 0.780 | 0.934 0.896 0.867 | +0.033 .. +0.106 | 0.0010 | 0.0120 | 0.0010 | 0.0120 | 0.0276 |

Pooled (k=11 AUROC tasks): **+0.046** AUROC, equal-weight mean (95% CI +0.003 .. +0.089, t-test across tasks p = 0.0374). Random effects (DL) +0.039 (95% CI +0.006 .. +0.071, p = 0.0189; I^2 = 86%); 7/11 tasks positive (sign test p = 0.549, Wilcoxon p = 0.054).

Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):

| tasks included | k | pooled gain | 95% CI | p |
|---|---|---|---|---|
| all AUROC tasks | 11 | +0.046 | +0.003 .. +0.089 | 0.037 |
| TDBRAIN counted as one dataset | 10 | +0.039 | -0.005 .. +0.084 | 0.075 |
| without TDBRAIN MDD + ADHD | 9 | +0.031 | -0.015 .. +0.077 | 0.156 |
| without ADHD-200 | 10 | +0.041 | -0.005 .. +0.088 | 0.076 |
| without TDBRAIN MDD + ADHD and ADHD-200 | 8 | +0.023 | -0.026 .. +0.072 | 0.299 |
| leave out crc_baxter | 10 | +0.056 | +0.014 .. +0.097 | 0.015 |
| leave out ibd_pooled | 10 | +0.053 | +0.008 .. +0.098 | 0.027 |
| leave out plasma_ad | 10 | +0.054 | +0.009 .. +0.098 | 0.023 |
| leave out csf_ad | 10 | +0.047 | -0.001 .. +0.095 | 0.056 |
| leave out autism_abide | 10 | +0.051 | +0.005 .. +0.097 | 0.034 |
| leave out adhd_adhd200 | 10 | +0.041 | -0.005 .. +0.088 | 0.076 |
| leave out mdd_tdbrain | 10 | +0.042 | -0.005 .. +0.088 | 0.074 |
| leave out adhd_tdbrain | 10 | +0.037 | -0.005 .. +0.079 | 0.079 |
| leave out sz_aseeg | 10 | +0.038 | -0.006 .. +0.083 | 0.080 |
| leave out abnormal_tuab | 10 | +0.045 | -0.003 .. +0.093 | 0.061 |
| leave out helsinki | 10 | +0.043 | -0.004 .. +0.091 | 0.069 |

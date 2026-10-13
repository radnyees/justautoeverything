# Shortcut exploitation (protocol v1) -- a central finding

Protocol v1 exposed site/study and demographics (age, sex; ABIDE head motion) as covariates
and scored multi-site tasks with pooled AUROC. The loop found and used these shortcuts.
Protocol v2 (all reported results) removes demographics/motion, keeps site/study only for
harmonization and scores WITHIN site. Below: (a) what non-biological inputs alone achieve on
the same Eval sets, (b) the archived v1 runs rescored within site, (c) which v1 champions used
shortcut variables in code.

## (a) Shortcut-only logistic regression (fit on dev pool, Eval, mean over 3 splits)

| task | inputs | pooled AUROC (v1 metric) | within-site AUROC (v2 metric) |
|---|---|---|---|
| adhd_adhd200 | site/study one-hot | 0.702 | 0.500 |
| adhd_adhd200 | age + sex | 0.664 | 0.647 |
| adhd_adhd200 | site + demographics | 0.775 | 0.647 |
| autism_abide | site/study one-hot | 0.568 | 0.501 |
| autism_abide | age + sex + motion | 0.585 | 0.571 |
| autism_abide | site + demographics | 0.596 | 0.561 |
| ibd_pooled | site/study one-hot | 0.693 | 0.500 |
| csf_ad | site/study one-hot | 0.727 | 0.500 |
| csf_ad | age + sex | 0.670 | 0.522 |
| csf_ad | site + demographics | 0.691 | 0.472 |

## (b) Archived v1 runs rescored (Eval)

| task | split | seed pooled | seed within-site | v1 champion pooled | v1 champion within-site | v2 champion within-site (reported) |
|---|---|---|---|---|---|---|
| adhd_adhd200 | 1 | 0.611 | 0.453 | 0.813 | 0.731 | 0.717 |
| ibd_pooled | 1 | 0.798 | 0.733 | 0.800 | 0.727 | 0.662 |
| ibd_pooled | 2 | 0.793 | 0.712 | 0.804 | 0.719 | 0.778 |
| ibd_pooled | 3 | 0.779 | 0.717 | 0.804 | 0.753 | 0.766 |
| csf_ad | 1 | 0.955 | 0.952 | 0.962 | 0.983 | 0.927 |
| csf_ad | 2 | 0.924 | 0.913 | 0.958 | 0.977 | 0.952 |
| csf_ad | 3 | 0.913 | 0.874 | 0.977 | 0.983 | 0.984 |

## (c) Shortcut variables referenced in v1 champion code

* adhd_adhd200 s1: Dev 0.660 -> 0.769; references age, sex, site
* adhd_adhd200 s2: Dev 0.646 -> 0.795; references age, sex, site
* adhd_adhd200 s3: Dev 0.644 -> 0.787; references age, sex, site
* autism_abide s1: Dev 0.751 -> 0.790; references age, mean_fd, sex, site
* autism_abide s2: Dev 0.737 -> 0.780; references age, mean_fd, sex, site
* autism_abide s3: Dev 0.734 -> 0.789; references mean_fd, site
* ibd_pooled s1: Dev 0.776 -> 0.811; references study
* ibd_pooled s2: Dev 0.801 -> 0.825; references study
* ibd_pooled s3: Dev 0.799 -> 0.838; references study
* csf_ad s1: Dev 0.882 -> 0.928; references none
* csf_ad s2: Dev 0.883 -> 0.935; references site
* csf_ad s3: Dev 0.904 -> 0.943; references none

Reading: on ADHD-200, site + age + sex alone reach a pooled Eval AUROC of ~0.77 with no
imaging at all, close to the v1 champion (0.813); within site, demographics alone still give
~0.65 because ADHD and control groups differ in sex/age inside sites. Any imaging model's
within-site AUROC on ADHD-200 should therefore be read against this demographic floor.

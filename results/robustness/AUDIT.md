# Transduction audit of seeds and champions

The Eval protocol featurizes dev-pool and Eval items in one `featurize` call (labels are never
available to it) and passes the whole Eval batch to `fit_predict`. A solution could therefore
use unlabeled Eval-set statistics. For every seed and champion (12 tasks x 3 splits):
* **featurize batch-dependent**: do the dev-pool features change when Eval items are featurized
  with them? (0 = per-item, inductive featurize)
* **chunked**: Eval items arrive in 20 chunks (10 for the large tasks); each chunk is featurized
  with the dev pool only and predicted on its own. Equals the joint score for inductive pipelines.
* **ref-batch**: as chunked, but each chunk is predicted inside a reference batch = dev pool +
  chunk (only the chunk's outputs kept), so any test-batch statistic in `fit_predict` (e.g. rank
  averaging, test-set scaling) is computed from training-time data -- a deployable variant.
* static scan: rank-type outputs in `fit_predict`; cross-item operations inside `featurize`.

Seeds: 1 (1 = all seeds exactly inductive).
Champions with batch-dependent featurize: 12/36; with batch-relative (rank) outputs: 11/36; failed audits: 0.
Champion Eval, joint -> ref-batch: mean change -0.0006, min -0.0244; joint -> chunked: mean -0.0187, min -0.2900.

| task | split | arm | featurize batch-dep. | rank outputs | Eval joint | chunked | ref-batch |
|---|---|---|---|---|---|---|---|
| crc_baxter | 1 | champion | no | no | 0.8159 | 0.8159 | 0.8159 |
| crc_baxter | 2 | champion | no | yes | 0.7240 | 0.6841 | 0.7585 |
| crc_baxter | 3 | champion | no | yes | 0.6791 | 0.5663 | 0.6562 |
| ibd_pooled | 1 | champion | yes | yes | 0.6617 | 0.6333 | 0.6494 |
| ibd_pooled | 2 | champion | yes | yes | 0.7776 | 0.7428 | 0.7630 |
| ibd_pooled | 3 | champion | yes | yes | 0.7657 | 0.6524 | 0.7882 |
| plasma_ad | 1 | champion | yes | no | 0.9231 | 0.9060 | 0.9060 |
| plasma_ad | 2 | champion | no | yes | 0.7949 | 0.7949 | 0.7949 |
| plasma_ad | 3 | champion | no | no | 0.8803 | 0.8803 | 0.8803 |
| csf_ad | 1 | champion | yes | yes | 0.9267 | 0.6367 | 0.9022 |
| csf_ad | 2 | champion | yes | no | 0.9522 | 0.9356 | 0.9356 |
| csf_ad | 3 | champion | yes | no | 0.9844 | 0.9922 | 0.9922 |
| prism_auc | 1 | champion | no | no | 0.1818 | 0.1818 | 0.1818 |
| prism_auc | 2 | champion | yes | yes | 0.1678 | 0.1396 | 0.1683 |
| prism_auc | 3 | champion | no | yes | 0.1312 | 0.1312 | 0.1312 |
| autism_abide | 1 | champion | no | no | 0.7468 | 0.7571 | 0.7609 |
| autism_abide | 2 | champion | yes | no | 0.8165 | 0.8175 | 0.8175 |
| autism_abide | 3 | champion | no | no | 0.7721 | 0.7649 | 0.7691 |
| adhd_adhd200 | 1 | champion | yes | no | 0.7169 | 0.7125 | 0.7125 |
| adhd_adhd200 | 2 | champion | no | no | 0.6292 | 0.6292 | 0.6292 |
| adhd_adhd200 | 3 | champion | no | no | 0.6493 | 0.6493 | 0.6493 |
| mdd_tdbrain | 1 | champion | no | no | 0.9844 | 0.9844 | 0.9844 |
| mdd_tdbrain | 2 | champion | no | no | 0.9334 | 0.9334 | 0.9334 |
| mdd_tdbrain | 3 | champion | no | no | 0.9441 | 0.9441 | 0.9441 |
| adhd_tdbrain | 1 | champion | no | no | 0.8692 | 0.8692 | 0.8692 |
| adhd_tdbrain | 2 | champion | no | no | 1.0000 | 1.0000 | 1.0000 |
| adhd_tdbrain | 3 | champion | no | no | 0.8974 | 0.8974 | 0.8974 |
| sz_aseeg | 1 | champion | yes | no | 0.7436 | 0.7372 | 0.7372 |
| sz_aseeg | 2 | champion | no | no | 0.7949 | 0.7949 | 0.7949 |
| sz_aseeg | 3 | champion | yes | no | 0.8590 | 0.8846 | 0.8846 |
| abnormal_tuab | 1 | champion | no | no | 0.9307 | 0.9307 | 0.9307 |
| abnormal_tuab | 2 | champion | no | yes | 0.8985 | 0.8897 | 0.8969 |
| abnormal_tuab | 3 | champion | no | yes | 0.9261 | 0.9166 | 0.9236 |
| helsinki | 1 | champion | no | no | 0.9342 | 0.9342 | 0.9342 |
| helsinki | 2 | champion | no | no | 0.8960 | 0.8960 | 0.8960 |
| helsinki | 3 | champion | no | no | 0.8666 | 0.8666 | 0.8666 |

(Seeds not listed are exactly inductive.) Interpretation: a champion whose ref-batch score
is at or above its joint score gains nothing from Eval-set statistics; a drop in the chunked
score alone means its outputs are batch-relative (ranks) and must be calibrated against a
reference cohort before per-patient use.

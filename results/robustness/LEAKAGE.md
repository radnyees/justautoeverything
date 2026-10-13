# Leakage checks (ids, files, network)

All runs below use network-isolated scoring containers (app `jae-sealed`, `block_network=True`).

## Positive controls (do the checks have power?)

* A solution that reads the label from the id (`':sz' in item`), sz_aseeg split 1, Eval AUROC: sealed, real ids **1.000**, sealed, order-preserving tokens **0.500**, sealed, random tokens **0.500**.
* Network probe inside the scoring container: **blocked**.

## Re-scoring every champion (and the seeds of tasks whose ids encode the label)

Three variants: real ids (determinism check); **order-preserving tokens** (the id TEXT is replaced but
the ids' sort order is kept, so anything that depends only on order -- e.g. sklearn GroupKFold's
assignment of `ctx['groups_train']` to inner-CV folds -- is unchanged); random tokens (the order
changes too). A solution that read labels from ids would lose performance under BOTH token
variants (cf. the control: 1.0 -> 0.5).

Largest absolute difference from the reported score (Eval and external sets): real ids **3.88e-04**; order-preserving tokens **3.88e-04**; random tokens **5.12e-02** (failed runs: sealed, real ids 0, sealed, order-preserving tokens 0, sealed, random tokens 0).

| task | split | arm | reported | real ids | order-preserving tokens | random tokens |
|---|---|---|---|---|---|---|
| abnormal_tuab | 1 | champion | 0.930644 | 0.930661 | 0.930644 | 0.930653 |
| abnormal_tuab | 2 | champion | 0.898547 | 0.898547 | 0.898547 | 0.898516 |
| abnormal_tuab | 3 | champion | 0.926046 | 0.926046 | 0.926046 | 0.926069 |
| adhd_adhd200 | 1 | champion | 0.716888 | 0.716888 | 0.716888 | 0.682774 |
| adhd_adhd200 | 2 | champion | 0.629236 | 0.629236 | 0.629236 | 0.629236 |
| adhd_adhd200 | 3 | champion | 0.649340 | 0.649340 | 0.649340 | 0.649340 |
| adhd_tdbrain | 1 | champion | 0.869231 | 0.869231 | 0.869231 | 0.869231 |
| adhd_tdbrain | 2 | champion | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| adhd_tdbrain | 3 | champion | 0.897436 | 0.897436 | 0.897436 | 0.897436 |
| autism_abide | 1 | champion | 0.746815 | 0.746815 | 0.746815 | 0.760533 |
| autism_abide | 2 | champion | 0.816538 | 0.816538 | 0.816538 | 0.816538 |
| autism_abide | 3 | champion | 0.772080 | 0.772080 | 0.772080 | 0.758401 |
| crc_baxter | 1 | champion | 0.815891 | 0.815891 | 0.815891 | 0.815891 |
| crc_baxter | 1 | seed | 0.854264 | 0.854264 | 0.854264 | 0.854264 |
| crc_baxter | 2 | champion | 0.724031 | 0.724419 | 0.724419 | 0.734109 |
| crc_baxter | 2 | seed | 0.755814 | 0.755814 | 0.755814 | 0.755814 |
| crc_baxter | 3 | champion | 0.679070 | 0.679070 | 0.679070 | 0.693411 |
| crc_baxter | 3 | seed | 0.774031 | 0.774031 | 0.774031 | 0.774031 |
| csf_ad | 1 | champion | 0.926667 | 0.926667 | 0.926667 | 0.926667 |
| csf_ad | 2 | champion | 0.952222 | 0.952222 | 0.952222 | 0.952222 |
| csf_ad | 3 | champion | 0.984444 | 0.984444 | 0.984444 | 0.984444 |
| helsinki | 1 | champion | 0.934188 | 0.934188 | 0.934188 | 0.934188 |
| helsinki | 2 | champion | 0.896044 | 0.896044 | 0.896044 | 0.896044 |
| helsinki | 3 | champion | 0.866611 | 0.866611 | 0.866611 | 0.866611 |
| ibd_pooled | 1 | champion | 0.661679 | 0.661679 | 0.661679 | 0.674144 |
| ibd_pooled | 2 | champion | 0.777583 | 0.777583 | 0.777583 | 0.777583 |
| ibd_pooled | 3 | champion | 0.765716 | 0.765716 | 0.765716 | 0.765716 |
| mdd_tdbrain | 1 | champion | 0.984375 | 0.984375 | 0.984375 | 0.984375 |
| mdd_tdbrain | 1 | seed | 0.870888 | 0.870888 | 0.870888 | 0.870888 |
| mdd_tdbrain | 2 | champion | 0.933388 | 0.933388 | 0.933388 | 0.933388 |
| mdd_tdbrain | 2 | seed | 0.791941 | 0.791941 | 0.791941 | 0.791941 |
| mdd_tdbrain | 3 | champion | 0.944079 | 0.944079 | 0.944079 | 0.944079 |
| mdd_tdbrain | 3 | seed | 0.887336 | 0.887336 | 0.887336 | 0.887336 |
| plasma_ad | 1 | champion | 0.923077 | 0.923077 | 0.923077 | 0.923077 |
| plasma_ad | 1 | seed | 0.940171 | 0.940171 | 0.940171 | 0.940171 |
| plasma_ad | 2 | champion | 0.794872 | 0.794872 | 0.794872 | 0.794872 |
| plasma_ad | 2 | seed | 0.888889 | 0.888889 | 0.888889 | 0.888889 |
| plasma_ad | 3 | champion | 0.880342 | 0.880342 | 0.880342 | 0.880342 |
| plasma_ad | 3 | seed | 0.905983 | 0.905983 | 0.905983 | 0.905983 |
| prism_auc | 1 | champion | 0.181776 | 0.181776 | 0.181776 | 0.189529 |
| prism_auc | 2 | champion | 0.167812 | 0.167812 | 0.167812 | 0.165849 |
| prism_auc | 3 | champion | 0.131237 | 0.131237 | 0.131237 | 0.136221 |
| sz_aseeg | 1 | champion | 0.743590 | 0.743590 | 0.743590 | 0.743590 |
| sz_aseeg | 1 | seed | 0.698718 | 0.698718 | 0.698718 | 0.698718 |
| sz_aseeg | 2 | champion | 0.794872 | 0.794872 | 0.794872 | 0.775641 |
| sz_aseeg | 2 | seed | 0.583333 | 0.583333 | 0.583333 | 0.583333 |
| sz_aseeg | 3 | champion | 0.858974 | 0.858974 | 0.858974 | 0.858974 |
| sz_aseeg | 3 | seed | 0.750000 | 0.750000 | 0.750000 | 0.750000 |

## Static scan of all saved solution code

1995 files (champions, best-of-20 and repeat candidates, archived runs). Lines touching files, network, subprocesses or the harness: 74; outside the resources a task provides (`ctx['genesets_dir']`, directly or via a helper function): **1**.
* `projects/microbiome/runs/ibd_pooled/s2/candidates/round_15.py:4`: `grouped inner CV scored as WITHIN-study AUROC (matching the harness metric)`

Limits: rejected candidates of the 36 main loops were not saved (only champions and the
round history), so they are covered only through their champions; the loops' scoring
containers (app `jae`) had network access and the data volume mounted -- the checks above show
that no saved solution used either.

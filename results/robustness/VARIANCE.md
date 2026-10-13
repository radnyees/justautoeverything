# Run-to-run variance (independent loops on the same split)

Split 1 of each cheaper task was re-run twice more from scratch (same seed solution, same Dev
CV, same Eval set; only the LLM's sampling differs): runs s1, s1r2, s1r3. Run-to-run SD is
the SD of the Eval gain over these 3 loops; split-to-split SD is over s1, s2, s3 (different
Eval sets).

| task | Eval gain s1 / s1r2 / s1r3 | run-to-run SD | split-to-split SD | Dev champion s1 / r2 / r3 |
|---|---|---|---|---|
| mdd_tdbrain | +0.113 / +0.116 / +0.096 | 0.011 | 0.043 | 0.912 / 0.912 / 0.930 |
| sz_aseeg | +0.045 / +0.038 / +0.013 | 0.017 | 0.084 | 0.906 / 0.918 / 0.884 |
| crc_baxter | -0.038 / -0.032 / +0.003 | 0.023 | 0.035 | 0.742 / 0.745 / 0.764 |
| ibd_pooled | +0.065 / +0.080 / +0.096 | 0.015 | 0.029 | 0.762 / 0.765 / 0.751 |
| plasma_ad | -0.017 / +0.000 / -0.026 | 0.013 | 0.042 | 0.942 / 0.943 / 0.937 |
| csf_ad | -0.026 / -0.030 / +0.048 | 0.044 | 0.068 | 0.946 / 0.943 / 0.923 |
| prism_auc | +0.014 / -0.013 / +0.016 | 0.016 | 0.002 | 0.143 / 0.142 / 0.143 |

Champion feature families per repeat (kept rounds):

* mdd_tdbrain s1: 7 kept / 20 rounds -- aperiodic/1-f-slope, artifact-handling, coherence/connectivity, complexity/fractal-dimension, linear-model, preprocessin
* mdd_tdbrain s1r2: 11 kept / 20 rounds -- aperiodic/1-f-slope, coherence/connectivity, complexity/fractal-dimension, covariance/riemannian, entropy, graph-theory,
* mdd_tdbrain s1r3: 10 kept / 20 rounds -- absolute-band-power, aperiodic/1-f-slope, artifact-handling, coherence/connectivity, complexity/fractal-dimension, covar
* sz_aseeg s1: 6 kept / 20 rounds -- aperiodic/1-f-slope, artifact-handling, complexity/fractal-dimension, covariance/riemannian, linear-model, microstates, 
* sz_aseeg s1r2: 9 kept / 20 rounds -- aperiodic/1-f-slope, artifact-handling, coherence/connectivity, covariance/riemannian, linear-model, microstates, peak-a
* sz_aseeg s1r3: 7 kept / 20 rounds -- coherence/connectivity, complexity/fractal-dimension, covariance/riemannian, ensemble/stacking, entropy, linear-model, m
* crc_baxter s1: 5 kept / 20 rounds -- ensemble/stacking, known-marker-taxa, linear-model, relative-abundance, taxonomic-aggregation, tree-ensemble
* crc_baxter s1r2: 6 kept / 20 rounds -- co-occurrence/network, feature-selection, known-marker-taxa, prevalence/abundance-filtering, relative-abundance, taxonom
* crc_baxter s1r3: 4 kept / 18 rounds -- alpha-diversity, ensemble/stacking, known-marker-taxa, linear-model, phylogeny/taxonomy-structure, prevalence/abundance-
* ibd_pooled s1: 7 kept / 20 rounds -- CLR/log-ratio, batch/location-correction, beta-diversity/ordination, ensemble/stacking, known-marker-taxa, linear-model,
* ibd_pooled s1r2: 8 kept / 20 rounds -- batch/location-correction, co-occurrence/network, ensemble/stacking, gradient-boosting, linear-model, relative-abundance
* ibd_pooled s1r3: 6 kept / 20 rounds -- CLR/log-ratio, alpha-diversity, batch/location-correction, ensemble/stacking, linear-model, phylogeny/taxonomy-structure
* plasma_ad s1: 6 kept / 20 rounds -- batch/site-correction, known-AD-biomarkers, log-transform, missing-value-imputation, missingness-as-signal, pathway/modu
* plasma_ad s1r2: 8 kept / 20 rounds -- batch/site-correction, ensemble/stacking, known-AD-biomarkers, log-transform, missing-value-imputation, pathway/module-a
* plasma_ad s1r3: 9 kept / 20 rounds -- batch/site-correction, dimensionality-reduction, ensemble/stacking, known-AD-biomarkers, log-transform, missing-value-im
* csf_ad s1: 10 kept / 20 rounds -- batch/site-correction, dimensionality-reduction, ensemble/stacking, known-AD-biomarkers, log-transform, missing-value-im
* csf_ad s1r2: 10 kept / 20 rounds -- batch/site-correction, ensemble/stacking, known-AD-biomarkers, log-transform, missing-value-imputation, pathway/module-a
* csf_ad s1r3: 7 kept / 20 rounds -- batch/site-correction, ensemble/stacking, gradient-boosting, known-AD-biomarkers, log-transform, missing-value-imputatio
* prism_auc s1: 9 kept / 20 rounds -- drug-similarity-sharing, drug-target-genes, ensemble/stacking, expression-normalization, gene-filtering/variance, kernel
* prism_auc s1r2: 7 kept / 20 rounds -- drug-similarity-sharing, ensemble/stacking, expression-normalization, gene-filtering/variance, moa-informed-features, mu
* prism_auc s1r3: 8 kept / 20 rounds -- drug-similarity-sharing, drug-target-genes, ensemble/stacking, gene-filtering/variance, kernel/GP, multi-task/shared-mod

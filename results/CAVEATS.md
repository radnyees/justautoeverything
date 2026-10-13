# Caveats and data notes (facts to account for when interpreting results)

## Protocol
- **Protocol v2 applies to every reported run.** v1 exposed demographics/site as covariates and
  used pooled AUROC; the loop exploited them (ADHD-200: site one-hot, sex, age, site x sex as
  predictors; pooled IBD: study one-hot). v1 runs of `adhd_adhd200`, `autism_abide` (stopped
  mid-run), `ibd_pooled` and `csf_ad` are archived under `projects/*/archive/` (with README) and
  are not part of the summary. Tasks whose v1 runs had no such inputs were not rerun:
  `crc_baxter`, `plasma_ad` (batch only, balanced 4 AD / 4 NC in the small batch), `prism_auc`
  (cancer lineage is a legitimate biological covariate).
- The v1 shortcut runs, the external-transfer results, the same-budget baselines, the corrected
  statistics, run-to-run variance and the transduction audit are reported as findings in
  `results/robustness/` (see its README), not only as caveats.
- **Selection on Dev.** Each loop keeps the best of ~20 candidates on a fixed 5-fold CV of the dev
  pool, so Dev gains are optimistically biased (winner's curse); only Eval (scored once) and the
  external datasets are unbiased. See fig1 (Dev vs Eval gain).
- **Candidate budget.** 1,500 s per candidate on 2/8/16 CPUs (task-dependent). Many failed
  rounds are timeouts of heavier ideas, not crashes (counts in `results/summary_runs.csv`).
- **LLM.** Claude Opus 5.5, effort `high`, Claude Code CLI 2.1.296 (orchestrated on Modal);
  earlier sandbox-run loops (microbiome CRC, proteomics plasma, drug response) used CLI
  2.1.287–2.1.296 with the same model and effort.

## Sample sizes (held-out Eval set per split ~ 25% of subjects)
- Small Eval sets -> wide CIs: plasma_ad (~22), sz_aseeg (~25), helsinki (20 neonates),
  csf_ad (~35), adhd_tdbrain (~40, age-matched adults only), mdd_tdbrain (70).
- Large: abnormal_tuab (688 recordings + 276 official eval), autism_abide (262), adhd_adhd200 (217),
  ibd_pooled (146), prism_auc (120 cell lines x 200 drugs).

## Task-specific
- **TDBRAIN (mdd, adhd).** Healthy controls in TDBRAIN were all >= 18 y; tasks are age-matched to
  the overlapping age range (ADHD therefore = adults only) and age is never given to the model.
  Controls and patients were recruited/recorded separately, so state/vigilance or protocol
  differences cannot be excluded. The seed's MDD signature is *inverted* on the external Mumtaz
  set (AUROC 0.25-0.29), consistent with dataset-specific signal; champions reach 0.28-0.62 there.
- **ASEEG.** Shipped preprocessed (1-45 Hz, 50 Hz notch, 128 Hz); the channel order is undocumented
  and was inferred from data (correlation-structure match to labelled RepOD recordings,
  r = 0.855 for standard 10-20 row order vs < 0.1 for alternatives;
  `projects/eeg_psych/data/aseeg_channel_order.py`). External sets: MSU (adolescents, 1 min,
  16 ch) and RepOD (14+14).
- **Mumtaz.** 4 EDF files in the public release are unreadable; 2 subjects lacking eyes-closed data
  are excluded from the external evaluation.
- **TUAB.** First 60 s skipped, next 5 min used, 19 channels, 100 Hz, clipped +/-800 uV
  (Gemein-style); task = official TRAIN split grouped by patient; official EVAL split is external.
- **Helsinki.** Label = majority (>= 2 of 3 experts) per second; metric = concatenated AUROC over
  held-out neonates. Literature numbers often use leave-one-out or other annotation rules.
- **ABIDE / ADHD-200.** Within-site AUROC (protocol v2) is stricter than the pooled accuracy/AUROC
  usually reported, so numbers are not directly comparable to the literature. ADHD-200 split 1's
  seed was below chance on its Eval set (0.453), which inflates that split's gain.
- **MicrobiomeHD Gevers.** The processed OTU table has only 162 stool samples (16 controls), so
  IBD is evaluated as a pooled four-study task (`ibd_pooled`); the single-study attempt is archived.
  **v3 (revision 2):** the v2 pooled task also contained 94 Huttenhower biopsies and 28 Engstrand
  tissue samples, and grouped by sample where a study had no subject column, so a person or co-twin
  could appear in both Dev and Eval. The task is now stool only and grouped by person (by twin pair
  in the Engstrand twin study); all IBD loops, baselines, repeats and audits were rerun, and the v2
  records are archived in `projects/microbiome/archive/v2_ibd_pooled_with_biopsies/`.
- **Proteomics.** Plasma: only open AD plasma MS set with labels (53 AD / 36 controls, one centre).
  CSF label = clinical diagnosis (the biochemical label is defined by CSF tau/amyloid).
- **Drug response.** 200 most variable drugs (>= 70% coverage) from PRISM secondary screen 19Q4;
  metric = mean per-drug Pearson r on held-out cell lines (pooled correlation would be inflated).

## Data availability
Raw data are not redistributed (TDBRAIN DUA, TUH/NEDC terms, CC BY-NC licences); every dataset is
re-downloadable with the scripts in `projects/*/data/`. Raw downloads were deleted from the Modal
volume after preprocessing; processed caches remain in the private `jae-data` volume.

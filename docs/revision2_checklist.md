# Revision 2 checklist (pre-submission review, 2026-10-13)

Each item: what is done, where the result lives, and how it was verified. Status: [ ] open, [x] done.

## A. Statistics and reporting
- [x] A1  Per-task inference that includes split (and repeat-run) variation: hierarchical bootstrap
      (resample splits, then subjects); report subject-conditional p, split-level t-test and
      hierarchical p side by side; say what each is conditional on. ADHD-200 footnote.
- [x] A2  p-values never shown as 0: (k+1)/(n+1) convention; Holm floor with 2,000 draws = 0.012.
- [x] A3  Sensitivity of the pooled gain: leave-one-task-out; without TDBRAIN (both tasks);
      without ADHD-200; without both; TDBRAIN counted as ONE dataset (shared controls).
- [x] A4  Absolute seed/champion scores next to every gain (seed-below-chance splits flagged).
- [x] A5  Optuna described as "search space hand-designed after, and informed by, the loop's
      findings" everywhere (README, BASELINES, figure legends).

## B. Confounds and leakage
- [x] B1  Demographic confounds (age, sex) for ADHD-200, ABIDE, TDBRAIN MDD/ADHD, TUAB:
      demographics-only Eval AUROC; do seed/champion predictions track age/sex within site;
      demographic-adjusted AUROC and likelihood-ratio test of the prediction given age+sex(+site).
      ASEEG has no demographics (state it).
- [x] B2  Pooled IBD: body sites and repeated samples per patient in each of the 4 studies;
      fix + rerun if patients can cross Dev/Eval.
- [x] B3  Labels encoded in subject IDs: harness ID-anonymization mode; re-score seeds and
      champions of affected tasks (sz_aseeg + MSU/RepOD, mdd_tdbrain + Mumtaz, plasma_ad,
      crc_baxter + external cohorts) and require identical scores.
- [x] B4  Scoring sandbox: evaluation containers without network; static scan of all saved
      code for file/network/harness access, written to a report.

## C. Variance
- [x] C1  Repeat loops (split 1, r2 + r3) for Helsinki and TUAB; Eval-scored; added to VARIANCE.

## D. Framing and disclosure
- [x] D1  External transfer as a finding: absolute AUROC vs chance (Mumtaz seed inverted, not a
      "gain"); TUAB official eval = same hospital (within-hospital check, not transfer);
      acquisition differences as an alternative explanation.
- [x] D2  Decisions made after Eval results were seen (v2 protocol, IBD redefinition, Mumtaz EC
      exclusion, Optuna spaces, pooled-estimator choice), with commit/time, in results/robustness/DISCLOSURES.md.
- [x] D3  Unequal conditions: v1 brief + earlier CLI for CRC/plasma/drug main loops; budget
      equalized in Dev evaluations only (not compute/tokens); early-stopped loops.
- [x] D4  Data-level choices on full data (drug selection by variance over all lines; proteomics
      normalized upstream), Helsinki metric protocol, seeds written by an LLM.

## E. Close-out
- [x] E1  Regenerate stats, tables, figures; re-read every headline number against the CSVs.
- [x] E2  Modal cost report (approved limit ~$160).

## Log
- A1/A3: hierarchical bootstrap, split t-test, sensitivity table and data-driven footnotes in core/stats.py
  (tested on adhd_adhd200 / crc_baxter / prism_auc); final run after the IBD rerun.
- A2: (k+1)/(n+1); Holm floor ~0.012. A4: absolute per-split scores in STATS.md.
- A5: README, core/search.py, BASELINES text, figure legend, template docstrings.
- B1: core/confounds_data.py (Modal) + core/confounds.py -> CONFOUNDS.md, fig13. TDBRAIN and ABIDE not
  explained by age/sex; ADHD-200 champion above the age+sex floor on split 1 only.
- B2: confirmed (94 Huttenhower biopsies, 28 Engstrand tissue samples, 18 people with >1 sample, 40 twin
  pairs). v3 task: stool only, person / twin-pair groups; verified no group crosses Dev/Eval
  (core/check_task.py). v2 archived (local == remote, 132 files). Rerun: loops, best-of-20, repeats,
  Optuna; then Eval scoring, audit, leakage check, shortcuts (done for v3).
- B3/B4: harness mode eval_anon; app jae-sealed (block_network); core/leakage_check.py with positive
  controls; static scan: 1,695 files, 73 file-access lines, all via ctx['genesets_dir'].
- D1: TRANSFER.md / fig11 rewritten (absolute AUROC vs chance, CIs, verdicts computed from data).
- D2-D4: results/robustness/DISCLOSURES.md (claims checked against git history and data).
- Process slips caught and fixed: logs/ missing after reset (job re-run), optuna missing after reset
  (job re-run), two unverified dataset descriptors and one wrong repeat-run claim removed.
- B2 done: IBD v3 loops/best-of-20/one-shot/Optuna/repeats scored; audit + leakage done. Loop gain
  v3: -0.041 / +0.093 / -0.011 (v2 with leakage: +0.065 / +0.031 / +0.008).
- B3/B4 done: 48 runs x 3 variants, no failures; order-preserving tokens == real ids for every run
  (max |diff| 3.9e-4 = numerical non-determinism, identical with real ids); random tokens up to 0.034
  (GroupKFold fold assignment of groups_train). Helsinki first failed on transient DNS when downloading
  large reports; rerun with score-only reports for tasks without external sets.
- C1: Helsinki repeats +0.055 / +0.051 (main +0.082); TUAB repeats running.
- C1 done: TUAB repeats +0.086 / +0.080 (main +0.091); Helsinki +0.055 / +0.051 (main +0.082).
  9 tasks now have 3 independent loops on split 1; run-to-run SD < split-to-split SD on 8/9.
- A1/A3 done: final stats with all repeats. Holm-significant (hierarchical): mdd_tdbrain, adhd_tdbrain,
  abnormal_tuab, helsinki. Conditional-only: crc_baxter (loss), adhd_adhd200. Pooled loop gain +0.050
  (p 0.029); without TDBRAIN +0.034 (p 0.13); without TDBRAIN and ADHD-200 +0.027 (p 0.25).
- E1 done: headline numbers recomputed independently from the raw final_scores.json files (separate
  code, not stats.py) -- all match the generated overview. Stale content found and fixed during the
  check: SHORTCUTS.md compared v1 IBD runs with the v3 champion (different Eval items) -> marked not
  comparable; fig11 title overstated ("performance does not carry over") -> "the loop's gains do not
  carry over"; static scan re-run over all 2,010 files (docstrings excluded): 0 lines outside the
  provided resources.
- E2 done: Modal $142.31 total (jae 10-12 $59.96, jae 10-13 $73.53, jae-sealed $8.82), approved ~$160.

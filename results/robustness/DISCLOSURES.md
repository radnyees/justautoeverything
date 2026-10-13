# Disclosures: analysis decisions, unequal conditions, data-level choices

Facts a reader needs to judge the results. Times are commit times (UTC, 2026-10-12/13) from the git
history of this repository.

## 1. Decisions made after some Eval results had been computed

| decision | when | what had been seen | consequence |
|---|---|---|---|
| Gevers-only IBD task replaced by pooled IBD (4 studies) | `251e072`, 10-12 07:54 | Eval of the Gevers task (16 stool controls: CIs of +-0.4) | archived in `projects/microbiome/archive/ibd_gevers_stool_only/` |
| Protocol v2 (no age/sex/motion; site/study only for harmonization; within-site AUROC) | `c7e876c`, 10-12 10:33 | v1 Eval for ADHD-200 (split 1), pooled IBD, CSF; v1 champion code using site/sex/age | v1 runs archived; v2 loops rerun on the **same split seeds 1-3**, i.e. on Eval sets whose v1 results had been seen |
| Mumtaz external set restricted to subjects with eyes-closed data (2 subjects excluded) | `dba77c5`, 10-12 12:08 (same commit as the v2 MDD final scores) | the record does not establish whether a Mumtaz score was seen first | 2 subjects; stated here |
| Optuna search spaces written | 10-13, after all loop results incl. Eval | loop champions' feature families | baseline is "loop-informed" (see BASELINES.md) |
| Pooled estimator: DerSimonian-Laird first, then equal-weight mean as primary | 10-13 | DL gave infinite weight to arms identical to the seed (SE = 0) | both are reported in `results/STATS.md` |
| Hierarchical bootstrap, sensitivity analyses, demographic and leakage checks | 10-13, revision 2 | all results | added as checks; per-task claims now use the hierarchical p |
| Pooled IBD v3 (stool only; grouped by person / twin pair) | 10-13, revision 2 | all v2 results (leak found by audit) | v2 archived in `projects/microbiome/archive/v2_ibd_pooled_with_biopsies/`; all IBD runs redone from scratch |

Every change listed above that altered a task re-ran that task's loops from scratch. A fully
pre-registered confirmation (new split seeds, analysis plan fixed in advance) was not done.

## 2. Conditions that differ between runs

* **Brief version.** The main loops of crc_baxter, plasma_ad and prism_auc started before protocol v2
  (10-12 06:44, 07:41, 08:33) and used the v1 wording of `core/brief.md` and of the microbiome /
  proteomics `program.md` (v1 described metadata as "site or age"; these tasks provide no age/sex
  either way). All baselines and all repeat loops used the v2 wording. Split-1 Eval gains, main
  loop (v1 wording) / two repeat loops (v2 wording): crc_baxter -0.038 / -0.032 / +0.003;
  plasma_ad -0.017 / +0.000 / -0.026; prism_auc +0.014 / -0.013 / +0.016 -- differences of the
  same size as the run-to-run spread on tasks without a wording change (VARIANCE.md), so the
  wording change cannot be separated from run-to-run variation.
* **Claude Code CLI.** 2.1.287-2.1.296 for the three early tasks (run in the sandbox); 2.1.296 for
  everything orchestrated on Modal. Model and effort identical (claude-opus-5-5, high).
* **Budget.** Arms are equalized in **Dev evaluations** (20), not in compute or LLM tokens: loop
  candidates can be heavier than Optuna trials; one-shot / best-of-20 / loop use LLM calls, Optuna
  none. The loop stops after 10 rounds without improvement: autism_abide split 1 used 19 rounds
  and split 2 used 11; all other loops and all best-of-20 runs used 20.
* **Scoring containers.** The loops' candidates were scored in containers with network access and
  the whole data volume mounted (labels included). No saved solution used either
  (`LEAKAGE.md`); network-isolated re-scoring with anonymized ids reproduces every champion score.

## 3. Data-level choices

* **Drug response:** the 200 drugs are the most variable (>= 70% coverage) across ALL cell lines,
  including later Eval lines (target selection on the full data; it affects seed and champion alike).
* **Proteomics:** intensities as released by the original studies, i.e. normalized by their
  pipelines across all samples.
* **Helsinki:** per-second labels by >= 2 of 3 experts and concatenated AUROC over held-out neonates;
  literature numbers often use leave-one-out and other annotation rules. Smoothing windows in the
  solutions are centred (non-causal), acceptable offline, not for online detection claims.
* **TDBRAIN:** the MDD and ADHD tasks share 90 healthy controls (of 128 and 119) and the recording
  setup, so they are not independent tasks (`results/STATS.md` sensitivity rows count them as one).
* **Subject ids that encode the label** (ASEEG `n`/`sz`, MSU `HE_`/`SZ_`, RepOD `h`/`s`, Mumtaz
  `HS`/`MDDS`, plasma `AD`/`NC`, two external CRC cohorts) were visible to solutions; no solution
  read them (LEAKAGE.md: static scan and anonymized re-scoring, with a positive control).
* **Demographics:** ASEEG and our TUAB build carry no age/sex, so their demographic confounding
  could not be checked.
* **Seeds** (the starting solutions) were written by the authors' coding assistant (Claude) before
  any loop ran; their strength is a design choice, which the same-budget baselines address.

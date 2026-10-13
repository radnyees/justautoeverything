# justautoeverything — LLM autoresearch loops for biomedical prediction

An **autoresearch loop** (in the spirit of Karpathy's *autoresearch* and the author's
[autoepi](https://github.com/radnyees/autoepi)) applied systematically to six biomedical
prediction problems / ten tasks. In every task an LLM (**Claude Opus 5.5, reasoning effort
`high`**) repeatedly rewrites one file, `solution.py`; a fixed harness scores it; the change is
kept only if it improves the development score. The question is *when* such a loop finds real,
transferable signal — and when it overfits or exploits shortcuts.

Results: [`results/SUMMARY.md`](results/SUMMARY.md) · figures: [`results/figures/`](results/figures/) ·
dataset and literature references: [`docs/references.md`](docs/references.md).

## Projects and tasks

| Project (folder) | Task | Data | Metric |
|---|---|---|---|
| `eeg_psych` | `mdd_tdbrain` | TDBRAIN V3.1 resting EEG, MDD vs healthy (age-matched); external: Mumtaz 2016 | AUROC |
| | `adhd_tdbrain` | TDBRAIN V3.1, ADHD/ADD vs healthy (age-matched) | AUROC |
| | `sz_aseeg` | ASEEG schizophrenia vs healthy; external: MSU, RepOD | AUROC |
| | `abnormal_tuab` | TUH Abnormal EEG v3.0.1 (train); external: official TUAB eval | AUROC |
| `fmri_connectivity` | `autism_abide` | ABIDE I preprocessed (PCP/CPAC), 20 sites | within-site AUROC |
| | `adhd_adhd200` | ADHD-200 preprocessed (Athena), 8 sites | within-site AUROC |
| `neonatal_seizure` | `helsinki` | Helsinki neonatal EEG, per-second detection | concatenated AUROC |
| `microbiome` | `crc_baxter` | MicrobiomeHD 16S, CRC vs healthy (Baxter); external: Zeller, Zackular, Xiang, Zhao | AUROC |
| | `ibd_pooled` | MicrobiomeHD 16S, IBD vs controls, 4 studies pooled | within-study AUROC |
| `proteomics_ad` | `plasma_ad` | Plasma DIA proteomics PXD065835, AD vs controls | AUROC |
| | `csf_ad` | CSF DIA proteomics (Bader 2020), clinical AD vs non-AD, 4 cohorts | within-site AUROC |
| `drug_response` | `prism_auc` | CCLE expression → PRISM secondary-screen AUC, 200 drugs | mean per-drug Pearson r |

## Protocol (identical for every task)

1. **Fixed harness, one editable file.** `core/harness.py` + each project's `prepare.py` define
   the data, labels, splits and metric and are never shown to or edited by the LLM. A solution
   implements `featurize(items, ctx)` (label-free) and `fit_predict(F_train, y_train, F_test, ctx)`;
   it never receives the labels of the items it is scored on.
2. **Splits.** Per split seed (1, 2, 3): 25% of subjects (grouped by subject/patient/neonate/cell
   line; stratified by label and site/study) form a held-out **Eval** set; the rest is the **dev
   pool**. The loop optimizes **Dev** = 5-fold grouped, stratified CV inside the dev pool.
3. **Loop.** `core/loop.py`, 20 rounds, patience 10, keep iff Dev improves by > 1e-4. Each round
   the LLM sees `core/brief.md`, the project's `program.md`, the task note, the current best
   solution and the full history (incl. the last failure), and returns one rationale, the
   feature/model families it uses, and a complete new `solution.py`. Every candidate is scored on
   Modal with a 1,500 s budget (timeouts/crashes count as failed rounds).
4. **Eval once.** After the loop, `core/final.py` fits the seed solution and the champion on the
   whole dev pool and scores each **once** on the held-out Eval set (and on external datasets);
   the champion-minus-seed difference is tested with a **paired, subject-clustered bootstrap**
   (2,000 resamples; one-sided p, 95% CI). Across the 3 splits we report mean ± sd and a t-test.
5. **Protocol v2 — no shortcuts.** Under v1, the loop learned recruitment/demographic shortcuts
   when they were available (ADHD-200: site, sex and age terms; pooled IBD: study identity). In v2
   no demographics (age, sex) or motion are provided; site/study/batch are provided **only for
   harmonization**, and multi-site tasks are scored **within site** (size-weighted mean of
   per-site AUROCs), so site prevalence cannot be exploited. Affected v1 runs are kept under
   `projects/*/archive/` with a README.
6. **Records.** Every round is logged in `projects/<p>/runs/<task>/s<seed>/history.jsonl`
   (rationale, families, score, kept, failure reason, model, effort, time);
   `baseline.json`, `solution.py` (champion), `final/` (Eval predictions) and `final_scores.json`
   sit next to it.

7. **Robustness analyses** (`results/robustness/`, figures 7-12; overview in
   `results/robustness/README.md`):
   * **Same-budget baselines** (same harness, splits, Dev CV, time limit, 20 Dev evaluations,
     keep-if-better on Dev, Eval scored once): *one-shot LLM* (round 1 of best-of-20), *best-of-20
     LLM without feedback* (`core/loop.py --strategy best_of_n`: every round sees only the brief
     and the seed), and *Optuna TPE* over a hand-written parameterized pipeline per project
     (`core/search.py`, `projects/*/baselines/{template,space}.py`; no LLM).
   * **Corrected statistics** (`core/stats.py`): task-level paired bootstrap that resamples
     subjects from the union of the 3 overlapping Eval sets; two-sided p with Holm and BH across
     the 12 tasks; pooled gain over the 11 AUROC tasks (equal-weight mean, t-interval across
     tasks; DerSimonian-Laird random effects as secondary).
   * **Run-to-run variance**: 2 extra independent loops on split 1 for 7 cheaper tasks
     (`runs_repeat/`, run names `s1r2`, `s1r3`).
   * **Shortcuts** (`core/shortcuts.py`): shortcut-only models (site/study, age/sex, motion) and the
     archived v1 champions rescored with the v2 within-site metric.
   * **Transduction audit** (`core/audit.py`, harness modes `audit`/`audit_ref`): do features
     depend on the batch, and does each seed/champion keep its Eval score when Eval items arrive
     in chunks (alone, or inside a dev-pool reference batch)?

## Layout

```
core/            harness, loop, Modal backend, final scoring, report, remote orchestration
projects/<p>/    prepare.py (fixed), project.json (tasks), program.md (LLM brief), seeds/,
                 data/ (download + preprocessing scripts), runs/ (all loop records), archive/,
                 baselines/ (Optuna template + search space), runs_bestof20/, runs_optuna/,
                 runs_repeat/ (baseline and repeat records, same layout as runs/)
results/         SUMMARY.md, summary_*.csv, STATS.md, stats_*.{csv,json}, robustness/, figures/
docs/            references.md (datasets, baselines, licences)
```

## Reproduce

```bash
# credentials outside the repo: ~/.config/autoresearch/{modal,claude}.env (see core/run.sh)
./core/run.sh modal deploy core/modal_app.py
./core/run.sh python core/prep.py projects/<p>/data/<build script>.py /data/<p>      # one-time data
./core/run.sh python core/remote.py launch <project> "<tasks>" "1 2 3"             # loops on Modal
./core/run.sh python core/remote.py sync <project>
./core/run.sh python core/final.py <project> <task>                                # Eval, once
python core/report.py                                                              # tables + figures
# robustness
./core/run.sh python core/remote.py launch <p> "<tasks>" "1 2 3" strategy=best_of_n subdir=runs_bestof20
./core/run.sh python core/remote.py launch <p> "<tasks>" "1" subdir=runs_repeat repeats="2 3"
./core/run_search_all.sh                                                           # Optuna baseline
./core/run_final_baselines.sh {bestof20|optuna|repeat} "<p>:<task> ..."            # Eval, once
./core/run_audit_all.sh                                                            # transduction audit
./core/run.sh python core/prep.py core/shortcuts.py                                # shortcut models
./core/run.sh python core/stats.py && python core/report_robustness.py
```

Data are not redistributed (several datasets forbid it, e.g. the TDBRAIN DUA and TUH/NEDC
terms); `projects/*/data/` contains the scripts that download and preprocess them from the
original sources. Compute: Modal (CPU containers, 2–16 cores); LLM: Claude Opus 5.5 via the
Claude Code CLI.

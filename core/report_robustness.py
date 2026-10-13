"""
core/report_robustness.py -- tables and figures for the robustness analyses (run after core/stats.py).

  results/robustness/BASELINES.md + baselines.csv   one-shot / best-of-20 / Optuna vs the loop (Eval)
  results/robustness/VARIANCE.md  + variance.csv    run-to-run (repeat loops, split 1) vs split-to-split
  results/robustness/SHORTCUTS.md                   v1 shortcut exploitation; shortcut-only models
  results/robustness/TRANSFER.md  + transfer.csv    in-distribution gain vs external-dataset gain
  results/robustness/AUDIT.md     + audit.csv       transduction audit of seeds and champions
  results/figures/fig7_baselines      fig8_meta_forest   fig9_variance
  results/figures/fig10_shortcuts     fig11_transfer     fig12_audit
    python core/report_robustness.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from report import GRID, INK, LABEL, MUT, PROJECT_ORDER, REF, color_of, plt, save  # noqa: E402
from metrics import within_site_auroc  # noqa: E402
from stats import TASKS, arm_report  # noqa: E402

RB = ROOT / "results" / "robustness"
ARM_STYLE = {"oneshot": ("One-shot LLM", "#9AA5B4", "v"), "bestof20": ("Best-of-20 LLM (no feedback)", "#D9860F", "s"),
             "optuna": ("Optuna, loop-informed space", "#1F9D55", "D"), "loop": ("Autoresearch loop", "#2563C9", "o")}
TASK_ORDER = [t for _, t in sorted(TASKS, key=lambda pt: (PROJECT_ORDER.index(pt[0]), TASKS.index(pt)))]
PROJ = dict((t, p) for p, t in TASKS)


def _fs(project, sub, task, run, tag=""):
    f = ROOT / "projects" / project / sub / task / run / f"final_scores{('_' + tag) if tag else ''}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _hist(project, sub, task, run):
    f = ROOT / "projects" / project / sub / task / run / "history.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ 1. baselines
def arm_scores():
    """Eval score per task x split x arm, keep-if-better-on-Dev gated (as the loop)."""
    rows = []
    for project, task in TASKS:
        for s in (1, 2, 3):
            seed = _fs(project, "runs", task, f"s{s}")
            if seed is None:
                continue
            rows.append({"project": project, "task": task, "split": s, "arm": "seed",
                         "eval": seed["eval_seed"], "dev": seed["dev_seed"], "adopted": None})
            for arm, (sub, tag) in {"loop": ("runs", ""), "bestof20": ("runs_bestof20", ""),
                                    "oneshot": ("runs_bestof20", "oneshot"), "optuna": ("runs_optuna", "")}.items():
                fs = _fs(project, sub, task, f"s{s}", tag)
                if fs is None:
                    continue
                failed = fs.get("failed") or fs.get("dev_champion") is None
                adopted = (not failed) and fs["dev_champion"] > fs["dev_seed"] + 1e-4
                rows.append({"project": project, "task": task, "split": s, "arm": arm,
                             "eval": fs["eval_champion"] if adopted else seed["eval_seed"],
                             "eval_raw": None if fs.get("failed") else fs.get("eval_champion"),
                             "dev": fs["dev_champion"] if adopted else seed["dev_seed"],
                             "dev_raw": fs.get("dev_champion"), "adopted": adopted,
                             "external": {k: v["champion"] for k, v in fs.get("external", {}).items()} if adopted
                             else {k: v["seed"] for k, v in seed.get("external", {}).items()}})
    return pd.DataFrame(rows)


def baselines(df, stats_df):
    d = df.copy()
    seed = d[d["arm"] == "seed"].set_index(["task", "split"])["eval"]
    d["gain"] = d.apply(lambda r: r["eval"] - seed.get((r["task"], r["split"]), np.nan), axis=1)
    d.drop(columns=["external"], errors="ignore").to_csv(RB / "baselines.csv", index=False, float_format="%.5f")
    arms = [a for a in ("oneshot", "bestof20", "optuna", "loop") if a in set(d["arm"])]
    piv = d.groupby(["task", "arm"])["eval"].mean().unstack()
    L = ["# Baselines with the same budget (Eval, mean over 3 splits)", "",
         "All arms start from the same seed solution, use the same harness, Dev CV, per-candidate time",
         "limit and the same 20 Dev evaluations, and are scored once on the same Eval sets:",
         "* **One-shot LLM** = round 1 of the best-of-20 run (one proposal; adopted only if it beats the",
         "  seed on Dev).",
         "* **Best-of-20 LLM** = 20 independent proposals, each seeing only the brief + the seed (no",
         "  history, no scores); best Dev candidate adopted if it beats the seed.",
         "* **Optuna, loop-informed space** = 20 Optuna TPE trials of a hand-written parameterized",
         "  pipeline per project (`projects/*/baselines/`); best Dev trial adopted if it beats the seed.",
         "  The optimizer uses no LLM, but the search spaces were written after the loop runs and include",
         "  feature families the loop's champions found (aperiodic slope, coherence, Hjorth parameters,",
         "  per-neonate normalization, ...). It is therefore informed by the loop's findings: a stronger",
         "  baseline than an independent search (conservative for the loop), not an LLM-free one.",
         "* **Loop** = the autoresearch loop (history + scores fed back each round).", "",
         "| task | seed | " + " | ".join(ARM_STYLE[a][0] for a in arms) + " |",
         "|---|---|" + "---|" * len(arms)]
    for t in TASK_ORDER:
        if t not in piv.index:
            continue
        cells = [f"{piv.loc[t, 'seed']:.3f}"] + [f"{piv.loc[t, a]:.3f}" if a in piv and pd.notna(piv.loc[t, a]) else "-"
                                               for a in arms]
        best = max((piv.loc[t, a], a) for a in arms if a in piv and pd.notna(piv.loc[t, a]))[1] if arms else None
        cells = [c if (i == 0 or arms[i - 1] != best) else f"**{c}**" for i, c in enumerate(cells)]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    one = d[d["arm"] == "oneshot"]
    if len(one):
        L += ["", f"One-shot proposals adopted on Dev: {int(one['adopted'].sum())}/{len(one)}; "
                  f"failed to run: {int(one['dev_raw'].isna().sum())}/{len(one)}."]
    if stats_df is not None and len(stats_df):
        L += ["", "## Paired tests (hierarchical bootstrap over splits, runs and subjects; Holm across 12 tasks)", "",
              "| comparison | tasks B > A | pooled gain (equal-weight, 11 AUROC tasks) | 95% CI | p | Holm-significant tasks |",
              "|---|---|---|---|---|---|"]
        pooled = json.loads((ROOT / "results" / "stats_pooled.json").read_text())
        for comp, g in stats_df.groupby("comparison", sort=False):
            p = pooled.get(comp)
            sig = ", ".join(f"{r.task} ({r.gain:+.3f})" for r in g.itertuples() if r.p_hier_holm < 0.05) or "none"
            if p:
                L.append(f"| {comp} | {int((g['gain'] > 0).sum())}/{len(g)} | {p['mean_equal']:+.3f} | "
                         f"{p['ci95_equal'][0]:+.3f} .. {p['ci95_equal'][1]:+.3f} | {_p(p['p_equal'])} | {sig} |")
    (RB / "BASELINES.md").write_text("\n".join(L) + "\n")

    # figure: mean gain over seed per task and arm, split points small
    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(TASK_ORDER) + 1.6))
    ys = {t: i for i, t in enumerate(TASK_ORDER[::-1])}
    off = {a: (k - (len(arms) - 1) / 2) * 0.17 for k, a in enumerate(arms)}
    for a in arms:
        name, c, m = ARM_STYLE[a]
        g = d[d["arm"] == a]
        for t, gg in g.groupby("task"):
            y = ys[t] + off[a]
            ax.scatter(gg["gain"], [y] * len(gg), s=12, color=c, alpha=0.45, marker=m, linewidth=0)
            ax.scatter([gg["gain"].mean()], [y], s=48, color=c, marker=m, edgecolor="white", linewidth=0.9,
                       zorder=3, label=name if t == g["task"].iloc[0] else None)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(list(ys.keys()), fontsize=8.5)
    for lab in ax.get_yticklabels():
        lab.set_color(color_of(PROJ[lab.get_text()]))
    ax.set_xlabel("Held-out Eval gain over the seed (large = mean of 3 splits, small = each split)\n"
                  "AUROC units; prism_auc in mean per-drug Pearson r")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Same budget (20 Dev evaluations): loop vs baselines", loc="left", fontweight="bold", pad=40)
    save(fig, "fig7_baselines")
    return d


def fig_meta(stats_df):
    pooled = json.loads((ROOT / "results" / "stats_pooled.json").read_text())
    comps = [c for c in ["seed->loop", "seed->bestof20", "seed->optuna", "bestof20->loop", "optuna->loop"]
             if c in set(stats_df["comparison"])]
    titles = {"seed->loop": "Loop vs seed", "seed->bestof20": "Best-of-20 vs seed", "seed->optuna": "Optuna (loop-informed) vs seed",
              "bestof20->loop": "Loop vs best-of-20", "optuna->loop": "Loop vs Optuna (loop-informed)"}
    fig, axes = plt.subplots(1, len(comps), figsize=(2.55 * len(comps) + 1.4, 0.36 * 13 + 1.6), sharey=True, sharex=True,
                             squeeze=False)
    order = TASK_ORDER[::-1]
    for ax, comp in zip(axes[0], comps):
        g = stats_df[stats_df["comparison"] == comp].set_index("task")
        for i, t in enumerate(order):
            if t not in g.index:
                continue
            r = g.loc[t]
            y = i + 1
            c = color_of(PROJ[t])
            ax.plot([r["ci_hier_lo"], r["ci_hier_hi"]], [y, y], color=c, lw=1.8,
                    alpha=0.9 if t != "prism_auc" else 0.5)
            sig = r["p_hier_holm"] < 0.05
            ax.scatter([r["gain"]], [y], s=34, color=c if sig else "white", edgecolor=c, linewidth=1.4, zorder=3)
        p = pooled.get(comp)
        if p:
            lo, hi, m = p["ci95_equal"][0], p["ci95_equal"][1], p["mean_equal"]
            ax.fill([lo, m, hi, m], [0, 0.28, 0, -0.28], color=INK, zorder=3)
            ax.text(0.98, 0.012, f"pooled {m:+.3f}\np={_p(p['p_equal'])}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7.5, color=INK)
        ax.axvline(0, color=REF, lw=1)
        ax.set_title(titles[comp], fontsize=9.5, loc="left", fontweight="bold")
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.7, len(order) + 0.7)
    axes[0][0].set_yticks(range(len(order) + 1))
    axes[0][0].set_yticklabels(["pooled (11 AUROC tasks)"] + order, fontsize=8)
    fig.supxlabel("Eval difference, mean over 3 splits; 95% CI from the hierarchical bootstrap (splits, runs, subjects). "
                  "Filled = Holm-adjusted hierarchical p < 0.05 across 12 tasks; diamond = equal-weight mean of the 11 AUROC tasks (t-interval)",
                  fontsize=8.5, color=MUT)
    fig.tight_layout()
    save(fig, "fig8_meta_forest")


# ------------------------------------------------------------------ 3. run-to-run variance
def variance():
    rows = []
    for project, task in TASKS:
        for run in ("s1", "s1r2", "s1r3", "s2", "s3"):
            sub = "runs_repeat" if "r" in run[1:] else "runs"
            fs = _fs(project, sub, task, run)
            if fs is None:
                continue
            h = _hist(project, sub, task, run)
            kept = [x for x in h if x["kept"]]
            fam = ", ".join(sorted({f for x in kept for f in (x.get("families") or [])}))[:120]
            rows.append({"project": project, "task": task, "run": run, "split": int(run[1]),
                         "repeat": int(run.split("r")[1]) if "r" in run[1:] else 1,
                         "dev_seed": fs["dev_seed"], "dev_champion": fs["dev_champion"], "eval_seed": fs["eval_seed"],
                         "eval_champion": fs["eval_champion"], "eval_gain": fs["eval_gain"], "rounds": len(h),
                         "kept": len(kept), "kept_families": fam})
    d = pd.DataFrame(rows)
    if d.empty or not (d["repeat"] > 1).any():
        return d
    d.to_csv(RB / "variance.csv", index=False, float_format="%.5f")
    rep_tasks = sorted(set(d[d["repeat"] > 1]["task"]), key=TASK_ORDER.index)
    L = ["# Run-to-run variance (independent loops on the same split)", "",
         "Split 1 of each cheaper task was re-run twice more from scratch (same seed solution, same Dev",
         "CV, same Eval set; only the LLM's sampling differs): runs s1, s1r2, s1r3. Run-to-run SD is",
         "the SD of the Eval gain over these 3 loops; split-to-split SD is over s1, s2, s3 (different",
         "Eval sets).", "",
         "| task | Eval gain s1 / s1r2 / s1r3 | run-to-run SD | split-to-split SD | Dev champion s1 / r2 / r3 |",
         "|---|---|---|---|---|"]
    for t in rep_tasks:
        g = d[d["task"] == t]
        rr = g[g["split"] == 1].sort_values("repeat")
        ss = g[g["repeat"] == 1].sort_values("split")
        L.append(f"| {t} | {' / '.join(f'{x:+.3f}' for x in rr['eval_gain'])} | {rr['eval_gain'].std(ddof=1):.3f} | "
                 f"{ss['eval_gain'].std(ddof=1):.3f} | {' / '.join(f'{x:.3f}' for x in rr['dev_champion'])} |")
    L += ["", "Champion feature families per repeat (kept rounds):", ""]
    for t in rep_tasks:
        for r in d[(d["task"] == t) & (d["split"] == 1)].sort_values("repeat").itertuples():
            L.append(f"* {t} {r.run}: {r.kept} kept / {r.rounds} rounds -- {r.kept_families or '-'}")
    (RB / "VARIANCE.md").write_text("\n".join(L) + "\n")

    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(rep_tasks) + 1.5))
    for i, t in enumerate(rep_tasks[::-1]):
        g = d[d["task"] == t]
        c = color_of(PROJ[t])
        rr, ss = g[g["split"] == 1], g[(g["repeat"] == 1) & (g["split"] > 1)]
        ax.scatter(rr["eval_gain"], [i + 0.12] * len(rr), s=40, color=c, edgecolor="white", linewidth=1, zorder=3,
                   label="split 1, 3 independent loops" if i == 0 else None)
        ax.scatter(ss["eval_gain"], [i - 0.12] * len(ss), s=36, facecolor="white", edgecolor=c, linewidth=1.4, zorder=3,
                   label="splits 2-3 (different Eval sets)" if i == 0 else None)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(range(len(rep_tasks)))
    ax.set_yticklabels(rep_tasks[::-1], fontsize=8.5)
    ax.set_xlabel("Held-out Eval gain over the seed")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=MUT, label="split 1, 3 independent loops"),
               plt.Line2D([0], [0], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUT,
                          label="splits 2-3 (different Eval sets)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Run-to-run vs split-to-split variation", loc="left", fontweight="bold", pad=26)
    save(fig, "fig9_variance")
    return d


# ------------------------------------------------------------------ 4a. shortcuts
V1 = {"adhd_adhd200": "projects/fmri_connectivity/archive/v1_demographics/adhd_adhd200",
      "autism_abide": "projects/fmri_connectivity/archive/v1_demographics/autism_abide",
      "ibd_pooled": "projects/microbiome/archive/v1_study_as_feature/ibd_pooled",
      "csf_ad": "projects/proteomics_ad/archive/v1_age_sex_site/csf_ad"}
SHORTCUT_RE = re.compile(r"""["'](site|sex|age|study|mean_fd|SITE_ID|Gender|Age)["']""")


def shortcuts():
    from sklearn.metrics import roc_auc_score
    raw = json.loads((RB / "shortcuts_raw.json").read_text())
    L = ["# Shortcut exploitation (protocol v1) -- a central finding", "",
         "Protocol v1 exposed site/study and demographics (age, sex; ABIDE head motion) as covariates",
         "and scored multi-site tasks with pooled AUROC. The loop found and used these shortcuts.",
         "Protocol v2 (all reported results) removes demographics/motion, keeps site/study only for",
         "harmonization and scores WITHIN site. Below: (a) what non-biological inputs alone achieve on",
         "the same Eval sets, (b) the archived v1 runs rescored within site, (c) which v1 champions used",
         "shortcut variables in code.", "",
         "## (a) Shortcut-only logistic regression (fit on dev pool, Eval, mean over 3 splits)", "",
         "| task | inputs | pooled AUROC (v1 metric) | within-site AUROC (v2 metric) |", "|---|---|---|---|"]
    sc_rows = []
    for t in ["adhd_adhd200", "autism_abide", "ibd_pooled", "csf_ad"]:
        r = raw[t]
        for k in r["kinds"]:
            po = np.mean([r["splits"][s][k]["pooled_auroc"] for s in r["splits"]])
            wi = np.mean([r["splits"][s][k]["within_site_auroc"] for s in r["splits"]])
            sc_rows.append({"task": t, "inputs": k, "pooled": po, "within": wi})
            name = {"site": "site/study one-hot", "demo": "age + sex" + (" + motion" if t == "autism_abide" else ""),
                    "site+demo": "site + demographics"}[k]
            L.append(f"| {t} | {name} | {po:.3f} | {wi:.3f} |")
    L += ["", "## (b) Archived v1 runs rescored (Eval)", "",
          "| task | split | seed pooled | seed within-site | v1 champion pooled | v1 champion within-site | "
          "v2 champion within-site (reported) |", "|---|---|---|---|---|---|---|"]
    v1_rows = []
    for t, d in V1.items():
        gs = raw[t]["group_site"]
        project = PROJ[t]
        for s in (1, 2, 3):
            f = ROOT / d / f"s{s}" / "final" / "champion.json"
            if not f.exists():
                continue
            out = {"task": t, "split": s}
            for arm in ("seed", "champion"):
                rep = json.loads((f.parent / f"{arm}.json").read_text())
                # pooled IBD: site = study = id prefix (v1/v2 ids include samples dropped in v3)
                sites = [str(g).split(":")[0] if t == "ibd_pooled" else gs[str(g)] for g in rep["groups"]]
                out[f"{arm}_pooled"] = roc_auc_score(rep["y"], rep["predictions"])
                out[f"{arm}_within"] = within_site_auroc(rep["y"], rep["predictions"], sites)
            v2 = _fs(project, "runs", t, f"s{s}")
            # pooled IBD was redefined in v3 (stool only, person/twin groups): its current champion is
            # scored on different Eval items, so it is not comparable with the v1 runs
            out["v2_within"] = (v2["eval_champion"] if v2 else np.nan) if t != "ibd_pooled" else np.nan
            v1_rows.append(out)
            L.append(f"| {t} | {s} | {out['seed_pooled']:.3f} | {out['seed_within']:.3f} | {out['champion_pooled']:.3f} | "
                     f"{out['champion_within']:.3f} | {out['v2_within']:.3f} |")
    L += ["", "nan = not comparable: pooled IBD was redefined in v3 (stool only, grouped by person / twin pair),",
          "so its current champion is scored on different Eval items; the v1 IBD runs used the v2-era data,",
          "which also contained biopsies and repeated samples (DISCLOSURES.md).",
          "", "## (c) Shortcut variables referenced in v1 champion code", ""]
    for t, d in V1.items():
        for s in (1, 2, 3):
            f = ROOT / d / f"s{s}" / "solution.py"
            if f.exists():
                used = sorted({m.group(1) for m in SHORTCUT_RE.finditer(f.read_text())})
                dev = (ROOT / d / f"s{s}" / "score.txt").read_text().strip() if (ROOT / d / f"s{s}" / "score.txt").exists() else "?"
                base = json.loads((ROOT / d / f"s{s}" / "baseline.json").read_text())["dev"] \
                    if (ROOT / d / f"s{s}" / "baseline.json").exists() else float("nan")
                L.append(f"* {t} s{s}: Dev {base:.3f} -> {float(dev):.3f}; references {', '.join(used) or 'none'}")
    L += ["", "Reading: on ADHD-200, site + age + sex alone reach a pooled Eval AUROC of ~0.77 with no",
          "imaging at all, close to the v1 champion (0.813); within site, demographics alone still give",
          "~0.65 because ADHD and control groups differ in sex/age inside sites. Any imaging model's",
          "within-site AUROC on ADHD-200 should therefore be read against this demographic floor."]
    (RB / "SHORTCUTS.md").write_text("\n".join(L) + "\n")

    # figure
    s = pd.DataFrame(sc_rows)
    v = pd.DataFrame(v1_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels, x = [], 0
    for t in ["adhd_adhd200", "autism_abide", "ibd_pooled", "csf_ad"]:
        for r in s[s["task"] == t].itertuples():
            c = color_of(PROJ[t])
            ax.bar(x - 0.2, r.pooled, 0.38, color=c, alpha=0.9)
            ax.bar(x + 0.2, r.within, 0.38, facecolor="white", edgecolor=c, hatch="////", linewidth=1.2)
            labels.append(f"{t}\n{r.inputs}")
            x += 1
        x += 0.5
    ax.set_xticks(_xticks(s))
    ax.set_xticklabels(labels, fontsize=6.6, rotation=60, ha="right")
    ax.axhline(0.5, color=REF, lw=1, ls="--")
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("Eval AUROC (no biological input)")
    ax.set_title("Shortcut-only models: pooled (solid) vs within-site (hatched)", loc="left", fontweight="bold",
                 fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax = axes[1]
    for i, r in enumerate(v.itertuples()):
        c = color_of(PROJ[r.task])
        ax.plot([i, i], [r.champion_within, r.champion_pooled], color=c, lw=1.5)
        ax.scatter([i], [r.champion_pooled], s=40, color=c, zorder=3, label="v1 champion, pooled" if i == 0 else None)
        ax.scatter([i], [r.champion_within], s=40, facecolor="white", edgecolor=c, linewidth=1.4, zorder=3,
                   label="v1 champion, within-site" if i == 0 else None)
        ax.scatter([i], [r.v2_within], s=46, marker="*", color=INK, zorder=4,
                   label="v2 champion, within-site" if i == 0 else None)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels([f"{r.task}\ns{r.split}" for r in v.itertuples()], fontsize=7, rotation=60, ha="right")
    ax.set_ylabel("Eval AUROC")
    ax.set_title("Archived v1 champions rescored", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig10_shortcuts")


def _xticks(s):
    xs, x = [], 0
    for t in ["adhd_adhd200", "autism_abide", "ibd_pooled", "csf_ad"]:
        for _ in range((s["task"] == t).sum()):
            xs.append(x)
            x += 1
        x += 0.5
    return xs


# ------------------------------------------------------------------ 4b. external transfer
EXT_KIND = {"tuab_official_eval": "same hospital (TUH) and protocol -- a within-hospital check, not transfer",
            "mumtaz": "different site and amplifier (adult MDD vs healthy)",
            "msu": "different site; adolescents; 1-min, 16-channel recordings",
            "repod": "different site (Warsaw); 14 + 14 adults",
            "crc_zeller": "different cohort and sequencing study", "crc_zackular": "different cohort and sequencing study",
            "crc_xiang": "different cohort and sequencing study", "crc_zhao": "different cohort and sequencing study"}


def _auc_ci(y, p, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            b.append(roc_auc_score(y[i], p[i]))
    return float(roc_auc_score(y, p)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _verdict(lo, hi):
    return "above chance" if lo > 0.5 else ("below chance" if hi < 0.5 else "not distinguishable from chance")


def transfer(arms_df):
    """External datasets: ABSOLUTE AUROC vs chance (95% subject-bootstrap CI) for seed and loop
    champion, next to in-distribution Eval. (A 'gain' over an inverted seed is not transfer.)"""
    rows = []
    for project, task in TASKS:
        for s in (1, 2, 3):
            run = ROOT / "projects" / project / "runs" / task / f"s{s}"
            fs = _fs(project, "runs", task, f"s{s}")
            if not fs or not fs.get("external"):
                continue
            reps = {a: json.loads((run / "final" / f"{a}.json").read_text()) for a in ("seed", "champion")}
            for name in fs["external"]:
                row = {"project": project, "task": task, "split": s, "external": name,
                       "internal_seed": fs["eval_seed"], "internal_loop": fs["eval_champion"]}
                for a, key in (("seed", "seed"), ("champion", "loop")):
                    e = reps[a]["external"][name]
                    auc, lo, hi = _auc_ci(e["y"], e["predictions"])
                    assert abs(auc - e["score"]) < 1e-9
                    row.update({f"ext_{key}": auc, f"ext_{key}_lo": lo, f"ext_{key}_hi": hi})
                row["verdict_loop"] = _verdict(row["ext_loop_lo"], row["ext_loop_hi"])
                for arm in ("bestof20", "optuna"):
                    r = arms_df[(arms_df["task"] == task) & (arms_df["split"] == s) & (arms_df["arm"] == arm)]
                    if len(r) and isinstance(r.iloc[0]["external"], dict):
                        row[f"ext_{arm}"] = r.iloc[0]["external"].get(name)
                rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RB / "transfer.csv", index=False, float_format="%.5f")
    L = ["# External transfer -- a finding: the loop's in-distribution gains do not carry over", "",
         "Seed and loop champion were fitted on the dev pool and applied ONCE, unchanged, to independent",
         "datasets. Shown are ABSOLUTE AUROCs with 95% subject-bootstrap CIs (2,000 draws) against chance",
         "(0.5) -- not 'gains', because a gain over a seed that is worse than chance (Mumtaz) only means",
         "moving toward chance. The TUAB official eval set comes from the same hospital and protocol as",
         "the training data, so it is a within-hospital check, not transfer. No harmonization across",
         "datasets was attempted; acquisition differences (amplifier, reference, montage, recording",
         "length, inferred ASEEG channel order, age range) are an alternative explanation to biology.", "",
         "Per task (computed from the table below):", ""]
    for (t, e), gg in d.groupby(["task", "external"], sort=False):
        n_above = int((gg["verdict_loop"] == "above chance").sum())
        n_better = int((gg["ext_loop"] > gg["ext_seed"]).sum())
        L.append(f"* {t} -> {e} ({EXT_KIND.get(e, '')}): loop champion above chance in {n_above}/{len(gg)} splits; "
                 f"better than the seed in {n_better}/{len(gg)}; mean external AUROC seed {gg['ext_seed'].mean():.3f} "
                 f"-> loop {gg['ext_loop'].mean():.3f} (in-distribution Eval of the loop: {gg['internal_loop'].mean():.3f}).")
    L += ["",
         "| task -> external | what differs | split | in-distribution Eval: seed -> loop | external AUROC seed [95% CI] | "
         "external AUROC loop [95% CI] | loop vs chance | best-of-20 | Optuna |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        b = getattr(r, "ext_bestof20", np.nan)
        o = getattr(r, "ext_optuna", np.nan)
        L.append(f"| {r.task} -> {r.external} | {EXT_KIND.get(r.external, '')} | {r.split} | "
                 f"{r.internal_seed:.3f} -> {r.internal_loop:.3f} | {r.ext_seed:.3f} [{r.ext_seed_lo:.2f}, {r.ext_seed_hi:.2f}] | "
                 f"{r.ext_loop:.3f} [{r.ext_loop_lo:.2f}, {r.ext_loop_hi:.2f}] | {r.verdict_loop} | "
                 f"{'-' if pd.isna(b) else f'{b:.3f}'} | {'-' if pd.isna(o) else f'{o:.3f}'} |")
    g = d.groupby(["task", "external"], sort=False).agg(
        int_loop=("internal_loop", "mean"), ext_seed=("ext_seed", "mean"), ext_loop=("ext_loop", "mean"),
        above=("verdict_loop", lambda v: int((v == "above chance").sum())), n=("verdict_loop", "size"))
    g["above"], g["n"] = g["above"].astype(int), g["n"].astype(int)
    L += ["", "## Summary (mean over 3 splits)", "",
          "| task -> external | in-distribution Eval (loop) | external AUROC seed -> loop | change vs in-distribution | "
          "splits with loop above chance |", "|---|---|---|---|---|"]
    for (t, e), r in g.iterrows():
        L.append(f"| {t} -> {e} | {r.int_loop:.3f} | {r.ext_seed:.3f} -> {r.ext_loop:.3f} | "
                 f"{r.ext_loop - r.int_loop:+.3f} | {r.above}/{r.n} |")
    (RB / "TRANSFER.md").write_text("\n".join(L) + "\n")

    # figure: in-distribution vs external AUROC, seed (hollow) and loop (filled), chance line
    keys = list(dict.fromkeys(zip(d["task"], d["external"])))
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(keys) + 1.6))
    for i, (t, e) in enumerate(keys[::-1]):
        g2 = d[(d.task == t) & (d.external == e)]
        c = color_of(PROJ[t])
        for r in g2.itertuples():
            y = i + (r.split - 2) * 0.16
            ax.plot([r.internal_loop, r.ext_loop], [y, y], color=c, lw=1, alpha=0.35)
            ax.plot([r.ext_loop_lo, r.ext_loop_hi], [y, y], color=c, lw=2.2, alpha=0.9, solid_capstyle="round")
            ax.scatter([r.internal_loop], [y], s=26, marker="|", color=INK, zorder=4)
            ax.scatter([r.ext_seed], [y], s=30, facecolor="white", edgecolor=c, linewidth=1.3, zorder=4)
            ax.scatter([r.ext_loop], [y], s=36, color=c, edgecolor="white", linewidth=0.8, zorder=5)
    ax.axvline(0.5, color=REF, lw=1.2, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{t} -> {e}" + ("  (same hospital)" if e == "tuab_official_eval" else "")
                        for t, e in keys[::-1]], fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("AUROC. Filled = loop champion on the external set (bar = 95% CI); hollow = seed;\n"
                  "| = loop champion on in-distribution Eval; dashed = chance. One row per split.", fontsize=8.5)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("External datasets: the loop's in-distribution gains do not carry over", loc="left",
                 fontweight="bold")
    save(fig, "fig11_transfer")


# ------------------------------------------------------------------ 5. transduction audit
def _seed_line(seeds):
    """One sentence on whether every audited seed scores identically when Eval is chunked."""
    diff = (seeds["eval_chunked"] - seeds["eval_joint"]).abs()
    n_bad = int((diff >= 1e-9).sum())
    if n_bad == 0:
        return f"Seeds: all {len(seeds)} are exactly inductive (chunked Eval score = joint Eval score)."
    return f"Seeds: {n_bad}/{len(seeds)} are not exactly inductive (chunked Eval score != joint Eval score)."


def audit():
    rows = []
    for project, task in TASKS:
        for s in (1, 2, 3):
            f = ROOT / "projects" / project / "runs" / task / f"s{s}" / "final" / "audit.json"
            if not f.exists():
                continue
            a = json.loads(f.read_text())
            for arm in ("seed", "champion"):
                if arm not in a or a[arm].get("failed"):
                    rows.append({"project": project, "task": task, "split": s, "arm": arm, "failed": True})
                    continue
                r, ref = a[arm], a.get(f"{arm}_ref", {})
                st = r.get("static", {})
                rows.append({"project": project, "task": task, "split": s, "arm": arm, "failed": False,
                             "featurize_batch_dependent": bool(r["feat_max_rel_diff"] > 1e-6),
                             "feat_max_rel_diff": r["feat_max_rel_diff"],
                             "batch_relative_outputs": bool(st.get("batch_relative_outputs")),
                             "static_featurize_hits": len(st.get("hits", [])),
                             "eval_joint": r["score_joint"], "eval_chunked": r["score"],
                             "eval_refbatch": ref.get("score", np.nan), "n_chunks": r["n_chunks"]})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    d.to_csv(RB / "audit.csv", index=False, float_format="%.5f")
    ok = d[~d["failed"]]
    ch = ok[ok["arm"] == "champion"]
    L = ["# Transduction audit of seeds and champions", "",
         "The Eval protocol featurizes dev-pool and Eval items in one `featurize` call (labels are never",
         "available to it) and passes the whole Eval batch to `fit_predict`. A solution could therefore",
         "use unlabeled Eval-set statistics. For every seed and champion (12 tasks x 3 splits):",
         "* **featurize batch-dependent**: do the dev-pool features change when Eval items are featurized",
         "  with them? (0 = per-item, inductive featurize)",
         "* **chunked**: Eval items arrive in 20 chunks (10 for the large tasks); each chunk is featurized",
         "  with the dev pool only and predicted on its own. Equals the joint score for inductive pipelines.",
         "* **ref-batch**: as chunked, but each chunk is predicted inside a reference batch = dev pool +",
         "  chunk (only the chunk's outputs kept), so any test-batch statistic in `fit_predict` (e.g. rank",
         "  averaging, test-set scaling) is computed from training-time data -- a deployable variant.",
         "* static scan: rank-type outputs in `fit_predict`; cross-item operations inside `featurize`.", "",
         _seed_line(ok[ok["arm"] == "seed"]),
         f"Champions with batch-dependent featurize: {int(ch['featurize_batch_dependent'].sum())}/{len(ch)}; with "
         f"batch-relative (rank) outputs: {int(ch['batch_relative_outputs'].sum())}/{len(ch)}; "
         f"failed audits: {int(d['failed'].sum())}.",
         f"Champion Eval, joint -> ref-batch: mean change {np.nanmean(ch['eval_refbatch'] - ch['eval_joint']):+.4f}, "
         f"min {np.nanmin(ch['eval_refbatch'] - ch['eval_joint']):+.4f}; joint -> chunked: mean "
         f"{np.nanmean(ch['eval_chunked'] - ch['eval_joint']):+.4f}, min {np.nanmin(ch['eval_chunked'] - ch['eval_joint']):+.4f}.",
         "", "| task | split | arm | featurize batch-dep. | rank outputs | Eval joint | chunked | ref-batch |",
         "|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        if r.failed:
            L.append(f"| {r.task} | {r.split} | {r.arm} | audit failed | | | | |")
            continue
        flag = r.featurize_batch_dependent or r.batch_relative_outputs or abs(r.eval_chunked - r.eval_joint) > 1e-9
        if r.arm == "seed" and not flag:
            continue
        L.append(f"| {r.task} | {r.split} | {r.arm} | {'yes' if r.featurize_batch_dependent else 'no'} | "
                 f"{'yes' if r.batch_relative_outputs else 'no'} | {r.eval_joint:.4f} | {r.eval_chunked:.4f} | "
                 f"{r.eval_refbatch:.4f} |")
    L += ["", "(Seeds not listed are exactly inductive.) Interpretation: a champion whose ref-batch score",
          "is at or above its joint score gains nothing from Eval-set statistics; a drop in the chunked",
          "score alone means its outputs are batch-relative (ranks) and must be calibrated against a",
          "reference cohort before per-patient use."]
    (RB / "AUDIT.md").write_text("\n".join(L) + "\n")

    ch = ch[ch["task"] != "prism_auc"]          # Pearson-r task: listed in AUDIT.md, off this AUROC plot
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    lo = float(np.nanmin(ch[["eval_joint", "eval_chunked", "eval_refbatch"]].values)) - 0.02
    hi = float(np.nanmax(ch[["eval_joint", "eval_chunked", "eval_refbatch"]].values)) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", color=REF, lw=1)
    for r in ch.itertuples():
        c = color_of(r.project)
        ax.plot([r.eval_joint, r.eval_joint], [r.eval_chunked, r.eval_refbatch], color=c, lw=1, alpha=0.5)
        ax.scatter([r.eval_joint], [r.eval_refbatch], s=36, color=c, edgecolor="white", linewidth=1, zorder=3)
        ax.scatter([r.eval_joint], [r.eval_chunked], s=30, facecolor="white", edgecolor=c, linewidth=1.3, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Champion Eval score, reported protocol (joint batch)")
    ax.set_ylabel("Inductive re-scoring: filled = ref-batch, hollow = chunked")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=color_of(p), label=LABEL[p]) for p in PROJECT_ORDER
               if p in set(ch["project"])]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Transduction audit: champions re-scored inductively (AUROC tasks)", loc="left",
                 fontweight="bold", fontsize=10)
    save(fig, "fig12_audit")


def overview():
    """results/robustness/README.md, generated: every number below is read from a result file."""
    st_ = pd.read_csv(ROOT / "results" / "stats_tasks.csv")
    pooled = json.loads((ROOT / "results" / "stats_pooled.json").read_text())
    names = {"seed->loop": "Autoresearch loop", "seed->optuna": "Optuna, loop-informed space",
             "seed->bestof20": "Best-of-20 LLM, no feedback", "seed->oneshot": "One-shot LLM"}
    L = ["# Robustness analyses -- overview (generated by core/report_robustness.py)", "",
         "Every number on this page is read from the result files listed in the table; Eval = held-out set,",
         "scored once. Per-task claims use the hierarchical bootstrap (splits, loop runs, subjects; Holm",
         "across 12 tasks); see results/STATS.md for what each p-value is conditional on.", "",
         "| analysis | files | figure |", "|---|---|---|",
         "| same-budget baselines | [BASELINES.md](BASELINES.md), baselines.csv | fig7_baselines |",
         "| corrected statistics, sensitivity | [../STATS.md](../STATS.md), ../stats_tasks.csv, ../stats_pooled.json | fig8_meta_forest |",
         "| run-to-run variance | [VARIANCE.md](VARIANCE.md), variance.csv | fig9_variance |",
         "| v1 shortcuts, shortcut-only models | [SHORTCUTS.md](SHORTCUTS.md), shortcuts_raw.json | fig10_shortcuts |",
         "| age/sex confounds | [CONFOUNDS.md](CONFOUNDS.md), confounds.csv | fig13_confounds |",
         "| external transfer | [TRANSFER.md](TRANSFER.md), transfer.csv | fig11_transfer |",
         "| transduction audit | [AUDIT.md](AUDIT.md), audit.csv | fig12_audit |",
         "| leakage (ids, files, network) | [LEAKAGE.md](LEAKAGE.md), leakage_rescore.csv | - |",
         "| decisions, unequal conditions, data choices | [DISCLOSURES.md](DISCLOSURES.md) | - |", "",
         "## Pooled Eval gain over the seed (11 AUROC tasks; equal-weight mean, t-interval across tasks)", "",
         "| arm (20 Dev evaluations) | pooled gain | 95% CI | p |", "|---|---|---|---|"]
    for comp, nm in names.items():
        if comp in pooled:
            p = pooled[comp]
            L.append(f"| {nm} | {p['mean_equal']:+.3f} | {p['ci95_equal'][0]:+.3f} .. {p['ci95_equal'][1]:+.3f} | "
                     f"{p['p_equal']:.3f} |")
    L += ["", "Head-to-head (B minus A): " + "; ".join(
        f"{c.replace('->', ' -> ')} {pooled[c]['mean_equal']:+.3f} (p = {pooled[c]['p_equal']:.3f})"
        for c in ("oneshot->loop", "bestof20->loop", "optuna->loop") if c in pooled) + "."]
    sens = pooled.get("seed->loop", {}).get("sensitivity", [])
    if sens:
        L += ["", "Sensitivity of the loop's pooled gain:", "", "| tasks included | k | gain | 95% CI | p |",
              "|---|---|---|---|---|"]
        for v in sens[:5]:
            L.append(f"| {v['label']} | {v['k']} | {v['mean']:+.3f} | {v['ci95'][0]:+.3f} .. {v['ci95'][1]:+.3f} | {v['p']:.3f} |")
    g = st_[st_["comparison"] == "seed->loop"].copy()
    sig = g[g["p_hier_holm"] < 0.05]
    cond_only = g[(g["p_holm"] < 0.05) & (g["p_hier_holm"] >= 0.05)]
    L += ["", "## Per task (loop vs seed)", "",
          "Holm-significant with the hierarchical bootstrap: " + (", ".join(
              f"{r.task} ({r.gain:+.3f}, p_Holm = {r.p_hier_holm:.3f})" for r in sig.itertuples()) or "none") + ".",
          "Significant only conditionally on the fitted models (subject bootstrap): " + (", ".join(
              f"{r.task} ({r.gain:+.3f})" for r in cond_only.itertuples()) or "none") + "."]
    v = RB / "variance.csv"
    if v.exists():
        d = pd.read_csv(v)
        rep = sorted(set(d[d["repeat"] > 1]["task"]))
        rr = {t: d[(d.task == t) & (d.split == 1)]["eval_gain"].std(ddof=1) for t in rep}
        ss = {t: d[(d.task == t) & (d.repeat == 1)]["eval_gain"].std(ddof=1) for t in rep}
        L += ["", "## Run-to-run variance", "",
              f"{len(rep)} tasks with 3 independent loops on split 1: run-to-run SD of the Eval gain "
              f"{min(rr.values()):.3f}-{max(rr.values()):.3f} (median {np.median(list(rr.values())):.3f}); "
              f"split-to-split SD {min(ss.values()):.3f}-{max(ss.values()):.3f} (median {np.median(list(ss.values())):.3f}); "
              f"run-to-run < split-to-split on {sum(rr[t] < ss[t] for t in rep)}/{len(rep)} tasks."]
    c = RB / "confounds.csv"
    if c.exists():
        d = pd.read_csv(c)
        L += ["", "## Age/sex confounds (mean over splits)", "", "| task | arm | AUROC | age+sex only | adjusted for age+sex |",
              "|---|---|---|---|---|"]
        for (t, a_), gg in d.groupby(["task", "arm"], sort=False):
            L.append(f"| {t} | {a_} | {gg.auc_reported.mean():.3f} | {gg.auc_demographics_only.mean():.3f} | "
                     f"{gg.auc_adjusted.mean():.3f} |")
    t = RB / "transfer.csv"
    if t.exists():
        d = pd.read_csv(t)
        L += ["", "## External transfer (loop champion, absolute AUROC; mean over splits)", "",
              "| task -> external | in-distribution | external seed -> loop | splits above chance | splits loop > seed |",
              "|---|---|---|---|---|"]
        for (tk, e), gg in d.groupby(["task", "external"], sort=False):
            L.append(f"| {tk} -> {e}{' (same hospital)' if e == 'tuab_official_eval' else ''} | {gg.internal_loop.mean():.3f} | "
                     f"{gg.ext_seed.mean():.3f} -> {gg.ext_loop.mean():.3f} | "
                     f"{int((gg.verdict_loop == 'above chance').sum())}/{len(gg)} | {int((gg.ext_loop > gg.ext_seed).sum())}/{len(gg)} |")
    au = RB / "audit.csv"
    if au.exists():
        d = pd.read_csv(au)
        ch = d[(d.arm == "champion") & (~d.failed)]
        dd = ch.eval_refbatch - ch.eval_joint
        w = ch.loc[dd.idxmin()]
        L += ["", "## Transduction audit", "",
              f"Champions: batch-dependent featurize {int(ch.featurize_batch_dependent.sum())}/{len(ch)}, batch-relative "
              f"(rank) outputs {int(ch.batch_relative_outputs.sum())}/{len(ch)}. Eval with a dev-pool reference batch minus "
              f"reported Eval: mean {dd.mean():+.4f}, worst {dd.min():+.4f} ({w.task} split {int(w.split)})."]
    lk = RB / "leakage_rescore.csv"
    if lk.exists():
        d = pd.read_csv(lk)
        L += ["", "## Leakage", "", "See LEAKAGE.md for the positive controls and per-run scores; largest |difference| "
              "from the reported Eval score: " + ", ".join(
                  f"{lab} {(d[m] - d['reported']).abs().max():.2e}" for m, lab in
                  (("eval", "real ids"), ("eval_anono", "order-preserving tokens"), ("eval_anon", "random tokens"))
                  if m in d) + "."]
    (RB / "README.md").write_text("\n".join(L) + "\n")


def main():
    RB.mkdir(parents=True, exist_ok=True)
    f = ROOT / "results" / "stats_tasks.csv"
    stats_df = pd.read_csv(f) if f.exists() else None
    arms = arm_scores()
    baselines(arms, stats_df)
    if stats_df is not None:
        fig_meta(stats_df)
    variance()
    shortcuts()
    transfer(arms)
    audit()
    if stats_df is not None:
        overview()
    print("wrote", RB)


if __name__ == "__main__":
    main()

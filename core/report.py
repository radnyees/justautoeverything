"""
core/report.py -- collect every finished run into paper-ready tables and figures.

Reads projects/<p>/runs/<task>/s<seed>/{baseline.json, history.jsonl, final_scores.json} and writes
  results/summary_runs.csv        one row per (project, task, split seed)
  results/summary_tasks.csv       one row per task: mean +/- sd over seeds, t-test, pooled CI
  results/SUMMARY.md              the same, as markdown tables
  results/figures/fig1_dev_vs_eval_gain.{png,pdf}   does a Dev gain carry over to held-out Eval?
  results/figures/fig2_eval_gain_forest.{png,pdf}   Eval gain + 95% CI per run, grouped by task
  results/figures/fig3_trajectories_<project>.{png,pdf}  best-so-far Dev per round, per task/seed
  results/figures/fig4_families.{png,pdf}           feature families tried / kept, per task
    python core/report.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"
INK, MUT, GRID, REF = "#141922", "#5B6675", "#EAEEF4", "#7C8797"
# categorical slots in fixed order (one per project)
PALETTE = ["#2563C9", "#D9860F", "#1F9D55", "#B5418E", "#7A5AF8", "#0E9AA7"]
PROJECT_ORDER = ["eeg_psych", "fmri_connectivity", "neonatal_seizure", "microbiome", "proteomics_ad",
                 "drug_response"]
LABEL = {"eeg_psych": "Resting EEG", "fmri_connectivity": "fMRI connectivity",
         "neonatal_seizure": "Neonatal seizures", "microbiome": "Gut microbiome",
         "proteomics_ad": "Proteomics (AD)", "drug_response": "Drug response"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#C6CFDC",
                     "axes.linewidth": 0.9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "figure.dpi": 110})


def load_runs():
    rows, hist = [], {}
    for proj_dir in sorted((ROOT / "projects").iterdir()):
        cfg_path = proj_dir / "project.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        for run in sorted(r for r in (proj_dir / "runs").glob("*/s*") if r.is_dir()) if (proj_dir / "runs").exists() else []:
            task, seed = run.parent.name, int(run.name[1:])
            h = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()] \
                if (run / "history.jsonl").exists() else []
            base = json.loads((run / "baseline.json").read_text()) if (run / "baseline.json").exists() else {}
            hist[(proj_dir.name, task, seed)] = (base.get("dev"), h)
            fs = json.loads((run / "final_scores.json").read_text()) if (run / "final_scores.json").exists() else None
            fails = Counter("timeout" if "TIMEOUT" in x.get("fail_log", "") else "crash"
                            for x in h if x["score"] in (None, float("-inf")))
            row = {"project": proj_dir.name, "task": task, "seed": seed,
                   "metric": cfg["tasks"].get(task, {}).get("metric", "AUROC"),
                   "rounds": len(h), "kept": sum(x["kept"] for x in h),
                   "failed_timeout": fails["timeout"], "failed_crash": fails["crash"],
                   "dev_seed": base.get("dev"), "dev_champion": max([base.get("dev") or -1] +
                                                                    [x["score"] for x in h if x["kept"]])}
            if fs:
                row.update(eval_seed=fs["eval_seed"], eval_champion=fs["eval_champion"],
                           eval_gain=fs["eval_gain"], ci_lo=fs["eval_bootstrap"]["ci95"][0],
                           ci_hi=fs["eval_bootstrap"]["ci95"][1], p_one_sided=fs["eval_bootstrap"]["p_one_sided"])
                for name, e in fs.get("external", {}).items():
                    row[f"ext:{name}"] = e["gain"]
            row["dev_gain"] = (row["dev_champion"] - row["dev_seed"]) if row["dev_seed"] is not None else None
            rows.append(row)
    return pd.DataFrame(rows), hist


def summarize(df):
    out = []
    for (p, t), g in df.dropna(subset=["eval_gain"]).groupby(["project", "task"]):
        gains = g["eval_gain"].values
        tt = stats.ttest_1samp(gains, 0) if len(gains) > 1 else None
        out.append({"project": p, "task": t, "metric": g["metric"].iloc[0], "n_splits": len(g),
                    "dev_seed": g["dev_seed"].mean(), "dev_champion": g["dev_champion"].mean(),
                    "dev_gain": g["dev_gain"].mean(), "eval_seed": g["eval_seed"].mean(),
                    "eval_champion": g["eval_champion"].mean(), "eval_gain_mean": gains.mean(),
                    "eval_gain_sd": gains.std(ddof=1) if len(gains) > 1 else np.nan,
                    "splits_improved": int((gains > 0).sum()),
                    "splits_sig_p05": int((g["p_one_sided"] < 0.05).sum()),
                    "t_test_p_two_sided": tt.pvalue if tt is not None else np.nan})
    return pd.DataFrame(out)


def color_of(project):
    return PALETTE[PROJECT_ORDER.index(project) % len(PALETTE)] if project in PROJECT_ORDER else REF


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_dev_vs_eval(df):
    d = df.dropna(subset=["eval_gain"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    lim = max(0.02, float(np.nanmax(np.abs(np.r_[d["dev_gain"], d["eval_gain"]]))) * 1.1)
    ax.axhline(0, color=REF, lw=1)
    ax.axvline(0, color=REF, lw=1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color=REF, lw=1, label="Eval gain = Dev gain")
    for p in PROJECT_ORDER:
        g = d[d["project"] == p]
        if len(g):
            ax.scatter(g["dev_gain"], g["eval_gain"], s=46, color=color_of(p), edgecolor="white",
                       linewidth=1.2, label=LABEL[p], zorder=3)
    ax.set_xlim(-0.02 * lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Dev gain (champion - seed, CV inside the dev pool)")
    ax.set_ylabel("Held-out Eval gain (scored once)")
    ax.set_title("Do loop gains on Dev carry over to held-out data?", loc="left", fontweight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8.5)
    save(fig, "fig1_dev_vs_eval_gain")


def fig_forest(df):
    d = df.dropna(subset=["eval_gain"]).copy()
    if d.empty:
        return
    d["key"] = d["project"].map(PROJECT_ORDER.index)
    d = d.sort_values(["key", "task", "seed"])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.4))
    ys = np.arange(len(d))[::-1]
    for y, (_, r) in zip(ys, d.iterrows()):
        c = color_of(r["project"])
        ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.scatter([r["eval_gain"]], [y], color=c, s=34, zorder=3, edgecolor="white", linewidth=1)
    ax.axvline(0, color=REF, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['task']}  s{r['seed']}" for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel("Held-out Eval gain over the seed, 95% CI (subject-clustered paired bootstrap)\n"
                  "AUROC units; prism_auc in mean per-drug Pearson r")
    ax.set_title("Eval gain per task and split", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], color=color_of(p), lw=3, label=LABEL[p])
               for p in PROJECT_ORDER if p in set(d["project"])]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
              fontsize=8.5)
    ax.set_title("Eval gain per task and split", loc="left", fontweight="bold", pad=34)
    save(fig, "fig2_eval_gain_forest")


def fig_trajectories(df, hist):
    for p in PROJECT_ORDER:
        keys = sorted(k for k in hist if k[0] == p)
        tasks = sorted({k[1] for k in keys})
        if not tasks:
            continue
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            for i, (_, _, s) in enumerate([k for k in keys if k[1] == t]):
                base, h = hist[(p, t, s)]
                if base is None:
                    continue
                best, xs, ys = base, [0], [base]
                for x in h:
                    if x["kept"]:
                        best = x["score"]
                    xs.append(x["round"])
                    ys.append(best)
                ax.step(xs, ys, where="post", lw=1.8, color=PALETTE[i % len(PALETTE)], label=f"split {s}")
            ax.set_title(t, loc="left", fontsize=10, fontweight="bold")
            ax.set_xlabel("round (0 = seed)")
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("best Dev score so far")
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{LABEL[p]}: loop trajectories (Opus 5.5, high effort)", x=0.01, ha="left",
                     fontweight="bold", y=1.04)
        save(fig, f"fig3_trajectories_{p}")


def fig_families(hist):
    """Which feature/model families end up in the FINAL champion? value = share of a task's split
    seeds whose champion (the last kept round) was tagged with the family by the LLM."""
    rows = []
    for (p, t, s), (_, h) in hist.items():
        kept = [x for x in h if x["kept"]]
        fams = {f.strip().lower() for f in (kept[-1].get("families", []) if kept else [])}
        rows.append({"task": t, "project": p, "seed": s, "fams": fams})
    if not rows:
        return
    d = pd.DataFrame(rows)
    tasks = sorted(d["task"].unique(), key=lambda t: (PROJECT_ORDER.index(d[d.task == t].project.iloc[0])
                                                       if d[d.task == t].project.iloc[0] in PROJECT_ORDER else 9, t))
    fams = sorted(set().union(*d["fams"]))
    M = pd.DataFrame(0.0, index=fams, columns=tasks)
    for t in tasks:
        g = d[d["task"] == t]
        for f in fams:
            M.loc[f, t] = np.mean([f in x for x in g["fams"]])
    M = M.loc[(M > 0).any(axis=1)]
    M = M.loc[M.sum(axis=1).sort_values(ascending=False).index[:40]]
    fig, ax = plt.subplots(figsize=(0.9 * M.shape[1] + 3.6, 0.27 * M.shape[0] + 1.6))
    im = ax.imshow(M.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=8)
    ax.set_title("Feature/model families in the final champions (share of splits)", loc="left",
                 fontweight="bold", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("share of the 3 splits", fontsize=8)
    save(fig, "fig4_families")


# published results on the SAME dataset and metric (others are not directly comparable)
LITERATURE = {"crc_baxter": (0.77, "Duvallet 2017 RF"), "helsinki": (0.955, "Tapani SVM (LOO)")}


def fig_before_after(df):
    """Per task: held-out Eval score of the seed (hollow) vs the champion (filled), per split."""
    d = df.dropna(subset=["eval_gain"]).copy()
    if d.empty:
        return
    d["key"] = d["project"].map(lambda p: PROJECT_ORDER.index(p) if p in PROJECT_ORDER else 9)
    tasks = d.sort_values(["key", "task"])["task"].drop_duplicates().tolist()
    n = len(tasks)
    cols = min(4, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.1 * cols, 2.7 * rows_), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        g = d[d["task"] == t].sort_values("seed")
        c = color_of(g["project"].iloc[0])
        for x, r in enumerate(g.itertuples()):
            ax.annotate("", xy=(x, r.eval_champion), xytext=(x, r.eval_seed),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.4, alpha=0.8))
            ax.scatter([x], [r.eval_seed], s=42, facecolor="white", edgecolor=c, linewidth=1.6, zorder=3)
            ax.scatter([x], [r.eval_champion], s=46, color=c, edgecolor="white", linewidth=1, zorder=4)
        if t in LITERATURE:
            v, lab = LITERATURE[t]
            ax.axhline(v, color=REF, ls="--", lw=1)
            ax.text(len(g) - 0.6, v, lab, ha="right", va="bottom", fontsize=7, color=MUT)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"split {s}" for s in g["seed"]], fontsize=8)
        ax.set_xlim(-0.6, len(g) - 0.4)
        ax.set_title(t, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylabel(g["metric"].iloc[0].replace("mean per-drug Pearson r", "mean drug r"), fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Held-out Eval: seed (hollow) -> loop champion (filled), scored once per split",
                 x=0.01, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_before_after")


def fig_external(df):
    """Seed vs champion on EXTERNAL datasets (other cohorts/sites), per split: do gains transfer?"""
    rows = []
    for run in (ROOT / "projects").glob("*/runs/*/s*/final_scores.json"):
        fs = json.loads(run.read_text())
        for name, e in fs.get("external", {}).items():
            rows.append({"project": fs["project"], "task": fs["task"], "seed": fs["seed"], "ext": name,
                         "seed_score": e["seed"], "champ": e["champion"],
                         "internal_gain": fs["eval_gain"]})
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values(["project", "task", "ext", "seed"])
    labels = [f"{r.task} -> {r.ext}  s{r.seed}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(d) + 1.6))
    ys = np.arange(len(d))[::-1]
    for y, r in zip(ys, d.itertuples()):
        c = color_of(r.project)
        ax.plot([r.seed_score, r.champ], [y, y], color=c, lw=1.6, alpha=0.6)
        ax.scatter([r.seed_score], [y], s=30, facecolor="white", edgecolor=c, linewidth=1.5, zorder=3)
        ax.scatter([r.champ], [y], s=34, color=c, edgecolor="white", linewidth=1, zorder=4)
    ax.axvline(0.5, color=REF, lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("External-dataset score (AUROC); hollow = seed, filled = champion; dashed = chance")
    ax.set_title("Transfer to external datasets (applied once, unchanged)", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "fig5_external_transfer")


def main():
    df, hist = load_runs()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "summary_runs.csv", index=False)
    tasks = summarize(df) if "eval_gain" in df else pd.DataFrame()
    tasks.to_csv(OUT / "summary_tasks.csv", index=False)
    md = ["# Results summary (auto-generated by core/report.py)\n",
          "Gains are champion minus seed. Eval = held-out 25% of subjects, scored once per split.\n"]
    if not tasks.empty:
        md.append(tasks.round(4).to_markdown(index=False))
    md.append("\n\n## Per run\n")
    md.append(df.round(4).to_markdown(index=False))
    (OUT / "SUMMARY.md").write_text("\n".join(md) + "\n")
    fig_dev_vs_eval(df)
    fig_forest(df)
    fig_trajectories(df, hist)
    fig_families(hist)
    fig_external(df)
    fig_before_after(df)
    print(tasks.round(3).to_string() if not tasks.empty else "no finished tasks yet")


if __name__ == "__main__":
    main()

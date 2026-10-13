"""
core/stats.py -- corrected statistics over the 12 tasks (uses only saved Eval predictions).

Task-level paired bootstrap: the three split seeds' Eval sets overlap (each is a random 25% of the
same subjects), so splits are not independent. Each bootstrap draw resamples SUBJECTS (groups) with
replacement from the union of the three Eval sets; every split is rescored on its own items of the
drawn subjects (with multiplicity), and the task statistic is the mean over splits of B - A.
  * two-sided p per task -> Holm (FWER) and Benjamini-Hochberg (FDR) across the 12 tasks;
  * pooled Eval gain over the 11 AUROC tasks (prism_auc, a Pearson-r task, is reported
    separately): PRIMARY = equal-weight mean of the task gains with a t-interval across tasks
    (tasks as the random unit; robust to arms that equal the seed, whose bootstrap SE is 0);
    SECONDARY = DerSimonian-Laird random effects (task SE floored at 0.005), fixed effect, sign
    test, Wilcoxon.
Comparisons (A -> B): seed -> loop, seed -> best-of-20, seed -> one-shot, seed -> Optuna,
best-of-20 -> loop, Optuna -> loop, one-shot -> loop. One-shot / Optuna / best-of-20 use the same
keep-if-better-on-Dev rule as the loop (a candidate replaces the seed only if its Dev score is higher).

    ./core/run.sh python core/stats.py            # writes results/stats_tasks.csv, stats_pooled.json, STATS.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from final import load_prepare  # noqa: E402

TASKS = [("microbiome", "crc_baxter"), ("microbiome", "ibd_pooled"), ("proteomics_ad", "plasma_ad"),
         ("proteomics_ad", "csf_ad"), ("drug_response", "prism_auc"), ("fmri_connectivity", "autism_abide"),
         ("fmri_connectivity", "adhd_adhd200"), ("eeg_psych", "mdd_tdbrain"), ("eeg_psych", "adhd_tdbrain"),
         ("eeg_psych", "sz_aseeg"), ("eeg_psych", "abnormal_tuab"), ("neonatal_seizure", "helsinki")]
NON_AUROC = {"prism_auc"}
SE_FLOOR = 0.005          # DL only: an arm identical to the seed has bootstrap SE 0 (infinite weight)
FLAT_AUROC = {t for _, t in TASKS} - {"prism_auc", "helsinki"}   # one scalar prediction per item
ARMS = {"seed": None, "loop": ("runs", "", "solution.py"), "bestof20": ("runs_bestof20", "", "solution.py"),
        "oneshot": ("runs_bestof20", "_oneshot", "candidates/round_01.py"),
        "optuna": ("runs_optuna", "", "solution.py")}
COMPARISONS = [("seed", "loop"), ("seed", "bestof20"), ("seed", "oneshot"), ("seed", "optuna"),
               ("bestof20", "loop"), ("optuna", "loop"), ("oneshot", "loop")]


def arm_report(project, task, s, arm):
    """Eval report (dict with predictions/y/groups/sites) of an arm on split s, or None if not
    available. Gated: an arm whose candidate did not beat the seed on Dev falls back to the seed."""
    pdir = ROOT / "projects" / project
    seed_f = pdir / "runs" / task / f"s{s}" / "final" / "seed.json"
    if arm == "seed":
        return json.loads(seed_f.read_text()) if seed_f.exists() else None
    sub, sfx, sol = ARMS[arm]
    run = pdir / sub / task / f"s{s}"
    fs = run / f"final_scores{sfx}.json"
    if not fs.exists():
        return None
    meta = json.loads(fs.read_text())
    if meta.get("failed") or meta.get("dev_champion") is None or meta["dev_champion"] <= meta["dev_seed"] + 1e-4:
        return json.loads(seed_f.read_text())          # not adopted on Dev -> the seed is kept
    return json.loads((run / f"final{sfx}" / "champion.json").read_text())


def _scorer(prep, task):
    import inspect
    four = len(inspect.signature(prep.score).parameters) >= 4

    def f(y, p, sites):
        return prep.score(y, p, task, sites) if (four and sites is not None) else prep.score(y, p, task)
    return f


def _auc(y, p):
    from scipy.stats import rankdata
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("one class")
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _fast_auroc(y, p, sites):
    """Vectorized AUROC / within-site AUROC (identical to sklearn / core.metrics.within_site_auroc)."""
    if sites is None:
        return _auc(y, p)
    num = den = 0.0
    for site in np.unique(sites):
        m = sites == site
        if 0 < y[m].sum() < m.sum():
            num += m.sum() * _auc(y[m], p[m])
            den += m.sum()
    if den == 0:
        raise ValueError("no site contains both classes")
    return num / den


def _pearson_cols(Yt, Yp):
    """Mean per-column Pearson r, exactly as projects/drug_response/prepare.py score (vectorized)."""
    M = np.isfinite(Yt) & np.isfinite(Yp)
    n = M.sum(0)
    X, Y = np.where(M, Yt, 0.0), np.where(M, Yp, 0.0)
    nn = np.maximum(n, 1)
    mx, my = X.sum(0) / nn, Y.sum(0) / nn
    dx, dy = np.where(M, Yt - mx, 0.0), np.where(M, Yp - my, 0.0)
    sxx, syy, sxy = (dx * dx).sum(0), (dy * dy).sum(0), (dx * dy).sum(0)
    ok = (n >= 10) & (sxx > 1e-24) & (syy > 1e-24)
    r = np.where(ok, sxy / np.sqrt(np.where(ok, sxx * syy, 1.0)), 0.0)
    return float(r.mean())


def _prep_special(task, a, b, gi):
    """Per-split closure idx -> (score_A, score_B) for nested-prediction tasks."""
    if task == "helsinki":                      # per-second labels concatenated over recordings
        lens = [len(np.asarray(v).ravel()) for v in a["y"]]
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        rec = [np.arange(st_, st_ + ln) for st_, ln in zip(starts, lens)]
        y = np.concatenate([np.asarray(v).ravel() for v in a["y"]]).astype(int)
        pa = np.concatenate([np.asarray(v, float).ravel() for v in a["predictions"]])
        pb = np.concatenate([np.asarray(v, float).ravel() for v in b["predictions"]])

        def f(idx):
            j = np.concatenate([rec[i] for i in idx])
            return _auc(y[j], pa[j]), _auc(y[j], pb[j])
        return f
    if task == "prism_auc":                     # rows = cell lines, columns = drugs
        Yt = np.asarray(a["y"], float)
        Pa = np.vstack([np.asarray(v, float) for v in a["predictions"]])
        Pb = np.vstack([np.asarray(v, float) for v in b["predictions"]])

        def f(idx):
            return _pearson_cols(Yt[idx], Pa[idx]), _pearson_cols(Yt[idx], Pb[idx])
        return f
    return None


def _task_bootstrap_special(task, pairs, n_boot, seed):
    union = sorted({str(g) for a, _ in pairs for g in a["groups"]})
    code = {g: i for i, g in enumerate(union)}
    prepped = []
    for a, b in pairs:
        gi = np.array([code[str(g)] for g in a["groups"]])
        prepped.append((_prep_special(task, a, b, gi), gi))

    def gain(counts=None):
        out = []
        for f, gi in prepped:
            idx = np.arange(len(gi)) if counts is None else np.repeat(np.arange(len(gi)), counts[gi])
            sa, sb = f(idx)
            out.append(sb - sa)
        return float(np.mean(out)), out
    return _boot_summary(gain, len(union), n_boot, seed)


def _boot_summary(gain, n_units, n_boot, seed):
    point, per_split = gain()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
        try:
            g, _ = gain(counts)
        except ValueError:
            continue
        if np.isfinite(g):
            boots.append(g)
    boots = np.array(boots)
    p_le, p_ge = (boots <= 0).mean(), (boots >= 0).mean()
    return {"gain": point, "per_split": per_split, "se": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "p_one_sided": float(p_le),
            "n_boot": int(len(boots)), "n_subjects": n_units}


def task_bootstrap(prep, task, pairs, n_boot=2000, seed=0):
    """pairs: list over splits of (report_A, report_B) on the same Eval items."""
    if task in FLAT_AUROC:
        return _task_bootstrap_fast(pairs, n_boot, seed)
    if task in ("helsinki", "prism_auc"):
        return _task_bootstrap_special(task, pairs, n_boot, seed)
    score = _scorer(prep, task)
    union = sorted({str(g) for a, _ in pairs for g in a["groups"]})
    code = {g: i for i, g in enumerate(union)}
    prepped = []
    for a, b in pairs:
        gi = np.array([code[str(g)] for g in a["groups"]])
        prepped.append((a["y"], a["predictions"], b["predictions"], a.get("sites"), gi))

    def gain(counts=None):
        out = []
        for y, pa, pb, sites, gi in prepped:
            if counts is None:
                idx = np.arange(len(gi))
            else:
                idx = np.repeat(np.arange(len(gi)), counts[gi])
            yy = [y[i] for i in idx]
            ss = None if sites is None else [sites[i] for i in idx]
            out.append(score(yy, [pb[i] for i in idx], ss) - score(yy, [pa[i] for i in idx], ss))
        return float(np.mean(out)), out

    point, per_split = gain()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union))
        try:
            g, _ = gain(counts)
        except ValueError:                      # a resample with one class only
            continue
        if np.isfinite(g):
            boots.append(g)
    boots = np.array(boots)
    p_le, p_ge = (boots <= 0).mean(), (boots >= 0).mean()
    return {"gain": point, "per_split": per_split, "se": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "p_one_sided": float(p_le),
            "n_boot": int(len(boots)), "n_subjects": len(union)}


def _task_bootstrap_fast(pairs, n_boot, seed):
    union = sorted({str(g) for a, _ in pairs for g in a["groups"]})
    code = {g: i for i, g in enumerate(union)}
    prepped = []
    for a, b in pairs:
        gi = np.array([code[str(g)] for g in a["groups"]])
        sites = None if a.get("sites") is None else np.asarray(a["sites"]).astype(str)
        prepped.append((np.asarray(a["y"]).astype(int), np.asarray(a["predictions"], float),
                        np.asarray(b["predictions"], float), sites, gi))

    def gain(counts=None):
        out = []
        for y, pa, pb, sites, gi in prepped:
            idx = np.arange(len(gi)) if counts is None else np.repeat(np.arange(len(gi)), counts[gi])
            ss = None if sites is None else sites[idx]
            out.append(_fast_auroc(y[idx], pb[idx], ss) - _fast_auroc(y[idx], pa[idx], ss))
        return float(np.mean(out)), out

    point, per_split = gain()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union))
        try:
            g, _ = gain(counts)
        except ValueError:
            continue
        boots.append(g)
    boots = np.array(boots)
    p_le, p_ge = (boots <= 0).mean(), (boots >= 0).mean()
    return {"gain": point, "per_split": per_split, "se": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "p_one_sided": float(p_le),
            "n_boot": int(len(boots)), "n_subjects": len(union)}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    run = 1.0
    for k, i in enumerate(order):
        run = min(run, p[i] * n / (n - k))
        adj[i] = run
    return adj


def dersimonian_laird(y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    w = 1 / v
    fe = float((w * y).sum() / w.sum())
    q = float((w * (y - fe) ** 2).sum())
    df = len(y) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c)
    wr = 1 / (v + tau2)
    re = float((wr * y).sum() / wr.sum())
    se_re = float(math.sqrt(1 / wr.sum()))
    z = re / se_re
    return {"k": len(y), "random_effects": re, "se": se_re, "ci95": [re - 1.96 * se_re, re + 1.96 * se_re],
            "z": z, "p_two_sided": float(2 * st.norm.sf(abs(z))), "tau2": tau2, "Q": q,
            "I2": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
            "fixed_effect": fe, "fixed_se": float(math.sqrt(1 / w.sum()))}


def _one(args):
    a_arm, b_arm, project, task = args
    pairs = []
    for s in (1, 2, 3):
        ra, rb = arm_report(project, task, s, a_arm), arm_report(project, task, s, b_arm)
        if ra is not None and rb is not None:
            pairs.append((ra, rb))
    if len(pairs) < 3:
        return None
    r = task_bootstrap(load_prepare(project), task, pairs, n_boot=2000)
    return {"comparison": f"{a_arm}->{b_arm}", "project": project, "task": task,
            **{k: r[k] for k in ("gain", "se", "p_two_sided", "p_one_sided", "n_boot", "n_subjects")},
            "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
            "split_gains": " ".join(f"{g:+.3f}" for g in r["per_split"])}


def main():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(a, b, p, t) for a, b in COMPARISONS for p, t in TASKS]
    with ProcessPoolExecutor(4) as ex:
        results = list(ex.map(_one, jobs))
    rows, pooled = [], {}
    for a_arm, b_arm in COMPARISONS:
        comp = f"{a_arm}->{b_arm}"
        comp_rows = [r for r in results if r is not None and r["comparison"] == comp]
        for r in comp_rows:
            print(f"{comp} {r['task']}: {r['gain']:+.4f} (95% CI {r['ci_lo']:+.3f}..{r['ci_hi']:+.3f}, "
                  f"p={r['p_two_sided']:.4f})", flush=True)
        if len(comp_rows) < 2:
            rows += comp_rows
            continue
        df = pd.DataFrame(comp_rows)
        df["p_holm"], df["p_bh"] = holm(df["p_two_sided"]), bh(df["p_two_sided"])
        rows += df.to_dict("records")
        au = df[~df["task"].isin(NON_AUROC)]
        se = np.maximum(au["se"].values, SE_FLOOR)
        dl = dersimonian_laird(au["gain"].values, se)
        g = au["gain"].values
        k = len(g)
        se_t = g.std(ddof=1) / math.sqrt(k)
        tq = st.t.ppf(0.975, k - 1)
        dl.update(mean_equal=float(g.mean()), se_equal=float(se_t),
                  ci95_equal=[float(g.mean() - tq * se_t), float(g.mean() + tq * se_t)],
                  p_equal=float(2 * st.t.sf(abs(g.mean() / se_t), k - 1)) if se_t > 0 else 1.0)
        npos = int((au["gain"] > 0).sum())
        dl.update(n_positive=npos, sign_test_p=float(st.binomtest(npos, len(au), 0.5).pvalue),
                  mean_unweighted=float(au["gain"].mean()),
                  wilcoxon_p=float(st.wilcoxon(au["gain"].values).pvalue) if len(au) >= 6 else None,
                  tasks=au["task"].tolist())
        pooled[comp] = dl
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "stats_tasks.csv", index=False, float_format="%.5f")
    (ROOT / "results" / "stats_pooled.json").write_text(json.dumps(pooled, indent=1) + "\n")
    write_md(out, pooled)


def _fmt_p(p):
    return "<0.0005" if p < 0.0005 else f"{p:.4f}"   # 2,000 bootstrap draws


def write_md(df, pooled):
    L = ["# Corrected statistics (Eval, 12 tasks)", "",
         "Task gain = mean over the 3 split seeds of (B - A) on the held-out Eval set. CI and p from a",
         "paired bootstrap that resamples subjects from the union of the 3 (overlapping) Eval sets",
         "(2,000 draws); p is two-sided. Holm = family-wise and BH = false-discovery-rate adjustment",
         "across the 12 tasks of each comparison. Pooled (11 AUROC tasks; prism_auc is mean per-drug",
         "Pearson r and is excluded): equal-weight mean of task gains with a t-interval across tasks",
         "(primary), plus DerSimonian-Laird random effects with task SE floored at 0.005 (secondary).",
         "Baseline arms use the loop's keep-if-better rule (a candidate replaces the seed only if its",
         "Dev score is higher), so every arm is a deployable pipeline chosen without Eval data.", ""]
    for comp, d in df.groupby("comparison", sort=False):
        L += [f"## {comp}", "", "| task | gain | 95% CI | p | p (Holm) | p (BH) | split gains |",
              "|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            L.append(f"| {r['task']} | {r['gain']:+.3f} | {r['ci_lo']:+.3f} .. {r['ci_hi']:+.3f} | "
                     f"{_fmt_p(r['p_two_sided'])} | {_fmt_p(r.get('p_holm', float('nan')))} | "
                     f"{_fmt_p(r.get('p_bh', float('nan')))} | {r['split_gains']} |")
        if comp in pooled:
            p = pooled[comp]
            L += ["", f"Pooled (k={p['k']} AUROC tasks): **{p['mean_equal']:+.3f}** AUROC, equal-weight mean "
                      f"(95% CI {p['ci95_equal'][0]:+.3f} .. {p['ci95_equal'][1]:+.3f}, t-test across tasks "
                      f"p = {p['p_equal']:.4f}). Random effects (DL) {p['random_effects']:+.3f} "
                      f"(95% CI {p['ci95'][0]:+.3f} .. {p['ci95'][1]:+.3f}, p = {p['p_two_sided']:.4f}; "
                      f"tau^2 = {p['tau2']:.4f}, I^2 = {100 * p['I2']:.0f}%); fixed effect "
                      f"{p['fixed_effect']:+.3f}; "
                      f"{p['n_positive']}/{p['k']} tasks positive (sign test p = {p['sign_test_p']:.3f}"
                      + (f", Wilcoxon p = {p['wilcoxon_p']:.3f}" if p.get("wilcoxon_p") is not None else "")
                      + ")."]
        L.append("")
    (ROOT / "results" / "STATS.md").write_text("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1:] == ["--md-only"]:            # rewrite STATS.md from the saved csv/json
        write_md(pd.read_csv(ROOT / "results" / "stats_tasks.csv"),
                 json.loads((ROOT / "results" / "stats_pooled.json").read_text()))
    else:
        main()

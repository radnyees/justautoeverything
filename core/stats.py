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
    p_le, p_ge = _boot_p(boots)
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
    p_le, p_ge = _boot_p(boots)
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
    p_le, p_ge = _boot_p(boots)
    return {"gain": point, "per_split": per_split, "se": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "p_one_sided": float(p_le),
            "n_boot": int(len(boots)), "n_subjects": len(union)}


def _boot_p(boots):
    """One-sided bootstrap p-values with the (k+1)/(n+1) convention (never 0; with 2,000 draws
    the smallest two-sided p is 2/2001 ~ 0.001, so the Holm floor over 12 tasks is ~0.012)."""
    n = len(boots)
    return (np.sum(boots <= 0) + 1) / (n + 1), (np.sum(boots >= 0) + 1) / (n + 1)


# ---------------------------------------------------------------- hierarchical bootstrap
def _pair_scorer(prep, task, a, b):
    """Positions in one split's Eval items -> (score_A, score_B)."""
    if task in FLAT_AUROC:
        y = np.asarray(a["y"]).astype(int)
        pa, pb = np.asarray(a["predictions"], float), np.asarray(b["predictions"], float)
        sites = None if a.get("sites") is None else np.asarray(a["sites"]).astype(str)

        def f(idx):
            ss = None if sites is None else sites[idx]
            return _fast_auroc(y[idx], pa[idx], ss), _fast_auroc(y[idx], pb[idx], ss)
        return f
    if task in ("helsinki", "prism_auc"):
        return _prep_special(task, a, b, None)
    score = _scorer(prep, task)

    def g(idx):
        yy = [a["y"][i] for i in idx]
        ss = None if a.get("sites") is None else [a["sites"][i] for i in idx]
        return (score(yy, [a["predictions"][i] for i in idx], ss),
                score(yy, [b["predictions"][i] for i in idx], ss))
    return g


def hier_bootstrap(prep, task, splits, n_boot=2000, seed=1):
    """splits: list over split seeds of (runs_A, runs_B), each a list of Eval reports (several only
    where independent repeat loops exist). Each draw resamples SPLITS with replacement, then (per
    drawn split) one run of each arm, then SUBJECTS (groups) within that split's Eval set; the
    statistic is the mean over the drawn splits of B - A. So the interval reflects split-to-split,
    run-to-run and subject sampling variation (with only 3 splits the split level is coarse)."""
    prepped = []
    for runs_a, runs_b in splits:
        g = np.asarray([str(x) for x in runs_a[0]["groups"]])
        for r in list(runs_a) + list(runs_b):
            assert [str(x) for x in r["groups"]] == list(g), "arms scored on different Eval items"
        ug, gi = np.unique(g, return_inverse=True)
        sc = {(i, j): _pair_scorer(prep, task, ra, rb) for i, ra in enumerate(runs_a) for j, rb in enumerate(runs_b)}
        prepped.append((sc, gi, len(ug), len(runs_a), len(runs_b)))
    point = []
    for sc, gi, ng, na, nb in prepped:
        sa, sb = sc[(0, 0)](np.arange(len(gi)))
        point.append(sb - sa)
    rng = np.random.default_rng(seed)
    boots = []
    K = len(prepped)
    for _ in range(n_boot):
        tot = []
        try:
            for k in rng.integers(0, K, K):
                sc, gi, ng, na, nb = prepped[k]
                counts = np.bincount(rng.integers(0, ng, ng), minlength=ng)
                idx = np.repeat(np.arange(len(gi)), counts[gi])
                sa, sb = sc[(int(rng.integers(na)), int(rng.integers(nb)))](idx)
                tot.append(sb - sa)
        except ValueError:                      # a resample with one class only
            continue
        boots.append(float(np.mean(tot)))
    boots = np.array(boots)
    p_le, p_ge = _boot_p(boots)
    return {"gain": float(np.mean(point)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "n_boot": int(len(boots)),
            "runs_per_split": [(na, nb) for _, _, _, na, nb in prepped]}


def run_reports(project, task, s, arm):
    """All available runs of an arm on split s: the primary run, plus independent repeat loops
    (runs_repeat/<task>/s<s>r<k>) for the loop arm."""
    first = arm_report(project, task, s, arm)
    if first is None:
        return []
    out = [first]
    if arm == "loop":
        for d in sorted((ROOT / "projects" / project / "runs_repeat" / task).glob(f"s{s}r*")):
            f = d / "final" / "champion.json"
            if (d / "final_scores.json").exists() and f.exists():
                out.append(json.loads(f.read_text()))
    return out


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


def _equal_weight(g):
    g = np.asarray(g, float)
    k = len(g)
    se = g.std(ddof=1) / math.sqrt(k)
    tq = st.t.ppf(0.975, k - 1)
    return {"k": k, "mean": float(g.mean()), "ci95": [float(g.mean() - tq * se), float(g.mean() + tq * se)],
            "p": float(2 * st.t.sf(abs(g.mean() / se), k - 1)) if se > 0 else 1.0}


TDBRAIN = ["mdd_tdbrain", "adhd_tdbrain"]


def sensitivity(g):
    """Pooled equal-weight gain under task exclusions. g: Series task -> gain (AUROC tasks).
    The two TDBRAIN tasks share control subjects and recording setup, so one variant counts them
    as a single dataset (their mean)."""
    out = [{"label": "all AUROC tasks", **_equal_weight(g)}]
    one = g.drop(TDBRAIN, errors="ignore")
    if all(t in g for t in TDBRAIN):
        one = pd.concat([one, pd.Series({"tdbrain (mean of MDD, ADHD)": g[TDBRAIN].mean()})])
    out.append({"label": "TDBRAIN counted as one dataset", **_equal_weight(one)})
    for label, drop in [("without TDBRAIN MDD + ADHD", TDBRAIN), ("without ADHD-200", ["adhd_adhd200"]),
                        ("without TDBRAIN MDD + ADHD and ADHD-200", TDBRAIN + ["adhd_adhd200"])]:
        out.append({"label": label, **_equal_weight(g.drop(drop, errors="ignore"))})
    for t in g.index:
        out.append({"label": f"leave out {t}", **_equal_weight(g.drop(t))})
    return out


def _one(args):
    a_arm, b_arm, project, task = args
    pairs = []
    for s in (1, 2, 3):
        ra, rb = arm_report(project, task, s, a_arm), arm_report(project, task, s, b_arm)
        if ra is not None and rb is not None:
            pairs.append((ra, rb))
    if len(pairs) < 3:
        return None
    prep = load_prepare(project)
    r = task_bootstrap(prep, task, pairs, n_boot=2000)
    splits = [(run_reports(project, task, s, a_arm), run_reports(project, task, s, b_arm)) for s in (1, 2, 3)]
    h = hier_bootstrap(prep, task, splits, n_boot=2000)
    assert abs(h["gain"] - r["gain"]) < 1e-9, "point estimates must agree"
    tt = st.ttest_1samp(r["per_split"], 0.0)
    absA, absB = [], []
    for ra, rb in pairs:
        sa, sb = _pair_scorer(prep, task, ra, rb)(np.arange(len(ra["groups"])))
        absA.append(sa)
        absB.append(sb)
        assert abs((sb - sa) - r["per_split"][len(absA) - 1]) < 1e-9
    return {"comparison": f"{a_arm}->{b_arm}", "project": project, "task": task,
            **{k: r[k] for k in ("gain", "se", "p_two_sided", "p_one_sided", "n_boot", "n_subjects")},
            "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
            "p_hier": h["p_two_sided"], "ci_hier_lo": h["ci95"][0], "ci_hier_hi": h["ci95"][1],
            "runs_split1": "x".join(str(v) for v in h["runs_per_split"][0]),
            "p_split_t": float(tt.pvalue),
            "abs_a": " ".join(f"{v:.3f}" for v in absA), "abs_b": " ".join(f"{v:.3f}" for v in absB),
            "split_gains": " ".join(f"{g:+.3f}" for g in r["per_split"])}


def main():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(a, b, p, t) for a, b in COMPARISONS for p, t in TASKS]
    import os
    with ProcessPoolExecutor(int(os.environ.get("JAE_STATS_WORKERS", "2"))) as ex:   # ~300 MB per worker
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
        df["p_hier_holm"], df["p_hier_bh"] = holm(df["p_hier"]), bh(df["p_hier"])
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
        dl["sensitivity"] = sensitivity(au.set_index("task")["gain"])
        pooled[comp] = dl
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "stats_tasks.csv", index=False, float_format="%.5f")
    (ROOT / "results" / "stats_pooled.json").write_text(json.dumps(pooled, indent=1) + "\n")
    write_md(out, pooled)


def _fmt_p(p):
    return "-" if p != p else f"{p:.4f}"


def _footnote(r):
    """Data-driven caveats for seed->loop rows (no hard-coded numbers)."""
    seeds = [float(v) for v in r["abs_a"].split()]
    gains = [float(v) for v in r["split_gains"].split()]
    out = []
    if r["task"] != "prism_auc" and min(seeds) < 0.6:
        k = int(np.argmin(seeds))
        out.append(f"{r['task']}: seed Eval on split {k + 1} was {seeds[k]:.3f} (below 0.6, the lowest of the 3 "
                   f"splits); split gains {r['split_gains']}, so the mean gain leans on that split.")
    if r["task"] == "adhd_adhd200":
        out.append("adhd_adhd200: age + sex alone give a comparable within-site AUROC; see "
                   "results/robustness/CONFOUNDS.md for the demographic floor per split.")
    if max(gains) > 0 > min(gains):
        out.append(f"{r['task']}: split gains change sign ({r['split_gains']}).")
    return " ".join(out)


def write_md(df, pooled):
    L = ["# Corrected statistics (Eval, 12 tasks)", "",
         "Task gain = mean over the 3 split seeds of (B - A) on each split's held-out Eval set (absolute",
         "scores of A and B per split are listed). Three p-values per task, all two-sided:", "",
         "* **p (subjects)** -- paired bootstrap resampling SUBJECTS from the union of the 3 overlapping Eval",
         "  sets. It is CONDITIONAL on the fitted models: it asks whether these particular champions beat",
         "  these seeds on new subjects from the same population, and ignores how much the result changes",
         "  with a different split or a different loop run.",
         "* **p (hierarchical)** -- bootstrap that resamples the 3 splits, then (where independent repeat",
         "  loops exist, split 1 of 9 tasks) the loop run, then subjects within the drawn split. It includes",
         "  split-to-split and run-to-run variation; with only 3 splits the split level is coarse.",
         "* **p (split t)** -- one-sample t-test of the 3 split gains (n = 3, low power).", "",
         "p-values use the (k+1)/(n+1) convention (2,000 draws), so the smallest two-sided p is ~0.001 and the",
         "smallest Holm-adjusted p over 12 tasks is ~0.012. Holm (family-wise) is applied within each",
         "comparison to both bootstrap p-values. Per-task claims should use the hierarchical p; tasks marked",
         "^ are significant only conditionally on the fitted models. Pooled (11 AUROC tasks; prism_auc is",
         "mean per-drug Pearson r): equal-weight mean of task gains, t-interval across tasks; DerSimonian-",
         "Laird random effects (task SE floored at 0.005) as secondary. Baseline arms use the loop's",
         "keep-if-better-on-Dev rule, so every arm is a pipeline chosen without Eval data.", ""]
    for comp, d in df.groupby("comparison", sort=False):
        a_arm, b_arm = comp.split("->")
        L += [f"## {comp}", "",
              f"| task | gain | {a_arm} (per split) | {b_arm} (per split) | 95% CI (hierarchical) | p (hierarchical) | "
              "Holm | p (subjects) | Holm | p (split t) |", "|---|---|---|---|---|---|---|---|---|---|"]
        notes = []
        for _, r in d.iterrows():
            flag = "^" if (r.get("p_holm", 1) < 0.05 and r.get("p_hier_holm", 1) >= 0.05) else ""
            fn = ""
            note = _footnote(r) if comp == "seed->loop" else ""
            if note:
                notes.append(note)
                fn = f" [{len(notes)}]"
            L.append(f"| {r['task']}{flag}{fn} | {r['gain']:+.3f} | {r['abs_a']} | {r['abs_b']} | "
                     f"{r['ci_hier_lo']:+.3f} .. {r['ci_hier_hi']:+.3f} | {_fmt_p(r['p_hier'])} | "
                     f"{_fmt_p(r.get('p_hier_holm', float('nan')))} | {_fmt_p(r['p_two_sided'])} | "
                     f"{_fmt_p(r.get('p_holm', float('nan')))} | {_fmt_p(r['p_split_t'])} |")
        if notes:
            L += [""] + [f"[{i}] {t}" for i, t in enumerate(notes, 1)]
        if comp in pooled:
            p = pooled[comp]
            L += ["", f"Pooled (k={p['k']} AUROC tasks): **{p['mean_equal']:+.3f}** AUROC, equal-weight mean "
                      f"(95% CI {p['ci95_equal'][0]:+.3f} .. {p['ci95_equal'][1]:+.3f}, t-test across tasks "
                      f"p = {p['p_equal']:.4f}). Random effects (DL) {p['random_effects']:+.3f} "
                      f"(95% CI {p['ci95'][0]:+.3f} .. {p['ci95'][1]:+.3f}, p = {p['p_two_sided']:.4f}; "
                      f"I^2 = {100 * p['I2']:.0f}%); {p['n_positive']}/{p['k']} tasks positive "
                      f"(sign test p = {p['sign_test_p']:.3f}"
                      + (f", Wilcoxon p = {p['wilcoxon_p']:.3f}" if p.get("wilcoxon_p") is not None else "") + ")."]
            sens = p.get("sensitivity") or []
            if sens:
                L += ["", "Sensitivity of the pooled gain (equal-weight mean, t-interval across tasks):", "",
                      "| tasks included | k | pooled gain | 95% CI | p |", "|---|---|---|---|---|"]
                for v in sens:
                    L.append(f"| {v['label']} | {v['k']} | {v['mean']:+.3f} | {v['ci95'][0]:+.3f} .. "
                             f"{v['ci95'][1]:+.3f} | {v['p']:.3f} |")
        L.append("")
    (ROOT / "results" / "STATS.md").write_text("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1:] == ["--md-only"]:            # rewrite STATS.md from the saved csv/json
        write_md(pd.read_csv(ROOT / "results" / "stats_tasks.csv"),
                 json.loads((ROOT / "results" / "stats_pooled.json").read_text()))
    else:
        main()

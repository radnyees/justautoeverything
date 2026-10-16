"""
core/probe.py -- TDBRAIN protocol probe, run and analysed exactly as pre-registered
(docs/preregistration_phase3.md, section 2). Never feeds back into any loop decision.

For mdd_tdbrain and adhd_tdbrain, probe sets P1 (technical only) and P2 (P1 + gross amplitude/EMG):
fit on each split's dev pool, score once on Eval (splits 1-3). Mean Eval AUROC over splits with a
bootstrap CI that resamples subjects from the union of the 3 (overlapping) Eval sets. Decision rule:
P1 positive iff mean AUROC >= 0.65 AND the CI excludes 0.5. Only if positive: seed and loop-champion Eval
predictions are additionally reported adjusted for the P1 probe score (residualization + LR test).
    ./core/run.sh python core/probe.py
Writes results/robustness/probe/*.json (Eval reports) and results/robustness/PROBE.md.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from loop import evaluate  # noqa: E402

TASKS = ["mdd_tdbrain", "adhd_tdbrain"]
SETS = ["P1", "P2"]
OUT = ROOT / "results" / "robustness" / "probe"
THRESH = 0.65


def run_probes():
    OUT.mkdir(parents=True, exist_ok=True)
    tpl = (ROOT / "projects" / "eeg_psych" / "probes" / "protocol_probe.py").read_text()
    jobs = {}
    with ThreadPoolExecutor(6) as ex:
        for t in TASKS:
            for ps in SETS:
                for s in (1, 2, 3):
                    f = OUT / f"{t}_{ps}_s{s}.json"
                    if f.exists():
                        continue
                    code = f"PARAMS = {{'set': '{ps}'}}\n" + tpl
                    jobs[f] = ex.submit(evaluate, "eeg_psych", t, "m", code, s, "eval", True, 1500)
        for f, fut in jobs.items():
            score, log, rep = fut.result()
            if rep is None:
                raise SystemExit(f"probe failed: {f.name}\n{log[-800:]}")
            f.write_text(rep if isinstance(rep, str) else json.dumps(rep))
            print(f"{f.stem}: Eval AUROC {score:.4f}", flush=True)


def _rep(path):
    r = json.loads(Path(path).read_text())
    return {"y": np.asarray(r["y"]).astype(int), "p": np.asarray(r["predictions"], float),
            "g": [str(x) for x in r["groups"]]}


def union_bootstrap(reps, n_boot=2000, seed=0):
    """Mean over splits of AUROC; CI by resampling subjects from the union of the Eval sets."""
    union = sorted({g for r in reps for g in r["g"]})
    code = {g: i for i, g in enumerate(union)}
    gi = [np.array([code[g] for g in r["g"]]) for r in reps]
    point = float(np.mean([roc_auc_score(r["y"], r["p"]) for r in reps]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union))
        vals = []
        for r, ix in zip(reps, gi):
            idx = np.repeat(np.arange(len(ix)), counts[ix])
            if 0 < r["y"][idx].sum() < len(idx):
                vals.append(roc_auc_score(r["y"][idx], r["p"][idx]))
        if len(vals) == len(reps):
            boots.append(np.mean(vals))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def adjusted(task, s, probe):
    """Seed / champion Eval predictions adjusted for the P1 probe score (same method as CONFOUNDS.md)."""
    out = {}
    for arm in ("seed", "champion"):
        r = _rep(ROOT / "projects" / "eeg_psych" / "runs" / task / f"s{s}" / "final" / f"{arm}.json")
        assert r["g"] == probe["g"], "probe and arm scored on different Eval items"
        z = (probe["p"] - probe["p"].mean()) / (probe["p"].std() + 1e-12)
        A = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(A, r["p"], rcond=None)
        resid = r["p"] - A @ beta
        pz = (r["p"] - r["p"].mean()) / (r["p"].std() + 1e-12)
        lr = 2 * (_loglik(np.column_stack([z, pz]), r["y"]) - _loglik(z[:, None], r["y"]))
        out[arm] = {"auc": float(roc_auc_score(r["y"], r["p"])), "auc_adjusted": float(roc_auc_score(r["y"], resid)),
                    "lr_p": float(st.chi2.sf(max(lr, 0), 1))}
    return out


def analyse():
    L = ["# TDBRAIN protocol probe (pre-registered: docs/preregistration_phase3.md, section 2)", "",
         "Can recording properties alone (not brain signals) separate patients from controls? Probe fitted",
         "on each split's dev pool, scored once on Eval; mean AUROC over the 3 splits, 95% CI from a bootstrap",
         "over subjects of the union of the Eval sets (2,000 draws).",
         f"Decision rule (fixed in advance): P1 positive iff mean AUROC >= {THRESH} and the CI excludes 0.5.", "",
         "| task | probe | Eval AUROC per split | mean | 95% CI | pre-registered verdict |", "|---|---|---|---|---|---|"]
    verdict = {}
    for t in TASKS:
        for ps in SETS:
            reps = [_rep(OUT / f"{t}_{ps}_s{s}.json") for s in (1, 2, 3)]
            per = [roc_auc_score(r["y"], r["p"]) for r in reps]
            m, lo, hi = union_bootstrap(reps)
            if ps == "P1":
                pos = m >= THRESH and (lo > 0.5 or hi < 0.5)
                verdict[t] = pos
                v = "POSITIVE (protocol signal)" if pos else "not positive"
            else:
                v = "descriptive only"
            L.append(f"| {t} | {ps} | {' / '.join(f'{x:.3f}' for x in per)} | {m:.3f} | {lo:.3f} .. {hi:.3f} | {v} |")
    L.append("")
    for t in TASKS:
        if verdict[t]:
            L += [f"## {t}: P1 positive -> predictions adjusted for the P1 probe score (pre-registered step)", "",
                  "| split | arm | AUROC | AUROC adjusted for probe | LR test p (prediction beyond probe) |",
                  "|---|---|---|---|---|"]
            for s in (1, 2, 3):
                a = adjusted(t, s, _rep(OUT / f"{t}_P1_s{s}.json"))
                for arm, v in a.items():
                    L.append(f"| {s} | {arm} | {v['auc']:.3f} | {v['auc_adjusted']:.3f} | {v['lr_p']:.3g} |")
            L += ["", f"Pre-registered consequence: {t} results are reported as potentially protocol-confounded.", ""]
        else:
            L += [f"{t}: no evidence that technical recording properties separate the groups (P1 not positive "
                  f"by the pre-registered rule; absence of evidence, limited by sample size).", ""]
    (ROOT / "results" / "robustness" / "PROBE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    run_probes()
    analyse()

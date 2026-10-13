"""
core/final.py -- one-shot final scoring of finished loops (run ONCE per run, after the loop).

For each split seed: the frozen seed solution and the champion are each fitted on the whole
dev pool and scored ONCE on the held-out Eval set (and on any external datasets), on Modal.
Predictions are saved, and the paired champion-minus-seed difference is tested with a
subject-clustered bootstrap. Nothing here feeds back into any loop decision.

    ./core/run.sh python core/final.py microbiome crc_baxter --seeds 1 2 3
Writes projects/<p>/runs/<task>/s<seed>/final/{seed,champion}.json and final_scores.json.

Baselines / repeats (same split seeds, same frozen seed predictions):
    final.py <p> <task> --subdir runs_bestof20                        # best-of-20 champions
    final.py <p> <task> --subdir runs_bestof20 --solution candidates/round_01.py --tag oneshot
    final.py <p> <task> --subdir runs_optuna
    final.py <p> <task> --subdir runs_repeat --runs s1r2 s1r3
The seed's Eval predictions are taken from runs/<task>/s<k>/final/seed.json when present, so every
comparison on a split is paired against identical seed predictions.
"""
import argparse
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bundle import bundle  # noqa: E402
from loop import project_cfg  # noqa: E402


def load_prepare(project):
    spec = importlib.util.spec_from_file_location("prepare_" + project,
                                                  ROOT / "projects" / project / "prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paired_bootstrap(score, y, p_seed, p_champ, groups, task, n_boot=2000, seed=0, sites=None):
    """Resample SUBJECTS with replacement; recompute champion-minus-seed metric each time
    (within-site metric when the task provides sites)."""
    if sites is not None:
        base_score = score
        score = lambda yy, pp, t, _s=None: base_score(yy, pp, t, _s)  # noqa: E731
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = [y[i] for i in idx]
        try:
            if sites is not None:
                ss = [sites[i] for i in idx]
                d = score(yy, [p_champ[i] for i in idx], task, ss) - score(yy, [p_seed[i] for i in idx], task, ss)
            else:
                d = score(yy, [p_champ[i] for i in idx], task) - score(yy, [p_seed[i] for i in idx], task)
        except ValueError:                      # resample with one class only
            continue
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    return {"n_boot": int(len(diffs)), "boot_se": float(diffs.std(ddof=1)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_one_sided": float((diffs <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("task")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--subdir", default="runs")
    ap.add_argument("--runs", nargs="+", default=None, help="run names (default s<seed>)")
    ap.add_argument("--solution", default="solution.py", help="file inside the run dir to score")
    ap.add_argument("--tag", default="", help="output suffix: final_scores<_tag>.json, final<_tag>/")
    ap.add_argument("--time-limit", type=int, default=3600)
    a = ap.parse_args()
    names = a.runs or [f"s{s}" for s in a.seeds]
    split_of = {n: int(re.match(r"s(\d+)", n).group(1)) for n in names}
    sfx = f"_{a.tag}" if a.tag else ""

    import modal
    cfg = project_cfg(a.project)["tasks"][a.task]
    fn = modal.Function.from_name("jae", f"ev_{cfg.get('size', 'm')}")
    prep = load_prepare(a.project)
    pdir = ROOT / "projects" / a.project
    seed_file = pdir / "seeds" / f"{a.task}.py"
    seed_code = (seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text()
    b = bundle(a.project)

    def seed_report(s):
        f = pdir / "runs" / a.task / f"s{s}" / "final" / "seed.json"
        return f.read_text() if f.exists() else None

    jobs = {}
    with ThreadPoolExecutor(8) as ex:
        for n in names:
            run = pdir / a.subdir / a.task / n
            s = split_of[n]
            if seed_report(s) is None and (s, "seed") not in jobs:
                jobs[(s, "seed")] = ex.submit(fn.remote, a.project, a.task, seed_code, "eval", s, True,
                                              a.time_limit, b)
            jobs[(n, "champion")] = ex.submit(fn.remote, a.project, a.task, (run / a.solution).read_text(),
                                              "eval", s, True, a.time_limit, b)
        res = {k: v.result() for k, v in jobs.items()}

    def report_of(key):
        r = res[key]
        if r.get("score") is None:
            return None, r.get("log", "")[-2000:]
        return r["report"], None

    for n in names:
        s = split_of[n]
        run = pdir / a.subdir / a.task / n
        (run / f"final{sfx}").mkdir(exist_ok=True)
        reps = {}
        txt = seed_report(s)
        if txt is None:
            txt, err = report_of((s, "seed"))
            if txt is None:
                raise SystemExit(f"{n} seed failed on Eval:\n{err}")
        reps["seed"] = json.loads(txt)
        if seed_report(s) is None or a.subdir == "runs":    # baselines reuse runs/<task>/s<k>/final/seed.json
            (run / f"final{sfx}" / "seed.json").write_text(txt)
        txt, err = report_of((n, "champion"))
        if txt is None:                         # e.g. a one-shot candidate that does not run
            out = {"project": a.project, "task": a.task, "seed": s, "run": n, "subdir": a.subdir,
                   "solution": a.solution, "failed": True, "error": err[-600:],
                   "eval_seed": reps["seed"]["score"]}
            (run / f"final_scores{sfx}.json").write_text(json.dumps(out, indent=2) + "\n")
            print(f"{a.project}/{a.task} {a.subdir}/{n} {a.solution}: FAILED on Eval", flush=True)
            continue
        reps["champion"] = json.loads(txt)
        (run / f"final{sfx}" / "champion.json").write_text(txt)
        base = json.loads((pdir / "runs" / a.task / f"s{s}" / "baseline.json").read_text()) \
            if not (run / "baseline.json").exists() else json.loads((run / "baseline.json").read_text())
        out = {"project": a.project, "task": a.task, "seed": s, "run": n, "subdir": a.subdir,
               "solution": a.solution, "dev_seed": base["dev"],
               "dev_champion": _dev_of(run, a.solution, base["dev"]),
               "eval_seed": reps["seed"]["score"], "eval_champion": reps["champion"]["score"]}
        out["dev_gain"] = (out["dev_champion"] - out["dev_seed"]) if out["dev_champion"] is not None else None
        out["eval_gain"] = out["eval_champion"] - out["eval_seed"]
        rs, rc = reps["seed"], reps["champion"]
        out["eval_bootstrap"] = paired_bootstrap(prep.score, rs["y"], rs["predictions"],
                                                 rc["predictions"], rs["groups"], a.task,
                                                 sites=rs.get("sites"))
        out["external"] = {}
        for name, ex_c in (rc.get("external") or {}).items():
            ex_s = rs["external"][name]
            e = {"seed": ex_s["score"], "champion": ex_c["score"],
                 "gain": ex_c["score"] - ex_s["score"]}
            e["bootstrap"] = paired_bootstrap(prep.score, ex_s["y"], ex_s["predictions"],
                                              ex_c["predictions"], ex_s["groups"], a.task,
                                              sites=ex_s.get("sites"))
            out["external"][name] = e
        (run / f"final_scores{sfx}.json").write_text(json.dumps(out, indent=2) + "\n")
        ext = "  ".join(f"{k} {v['seed']:.3f}->{v['champion']:.3f}" for k, v in out["external"].items())
        dev = f"{out['dev_champion']:.3f}" if out["dev_champion"] is not None else "?"
        print(f"{a.project}/{a.task} {a.subdir}/{n} {a.solution}: Dev {out['dev_seed']:.3f}->{dev} | "
              f"Eval {out['eval_seed']:.3f}->{out['eval_champion']:.3f} "
              f"(gain {out['eval_gain']:+.3f}, 95% CI {out['eval_bootstrap']['ci95'][0]:+.3f}.."
              f"{out['eval_bootstrap']['ci95'][1]:+.3f}, p={out['eval_bootstrap']['p_one_sided']:.3f})"
              + (f" | external: {ext}" if ext else ""), flush=True)


def _dev_of(run, solution, dev_seed):
    """Dev score of the scored file: score.txt for the champion, the history record for a candidate."""
    if solution == "solution.py":
        return float((run / "score.txt").read_text())
    m = re.match(r"candidates/round_(\d+)\.py$", solution)
    if m and (run / "history.jsonl").exists():
        for line in (run / "history.jsonl").read_text().splitlines():
            h = json.loads(line)
            if h["round"] == int(m.group(1)):
                return h["score"] if h["score"] is not None and np.isfinite(h["score"]) else None
    return None


if __name__ == "__main__":
    main()

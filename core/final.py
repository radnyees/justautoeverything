"""
core/final.py -- one-shot final scoring of finished loops (run ONCE per run, after the loop).

For each split seed: the frozen seed solution and the champion are each fitted on the whole
dev pool and scored ONCE on the held-out Eval set (and on any external datasets), on Modal.
Predictions are saved, and the paired champion-minus-seed difference is tested with a
subject-clustered bootstrap. Nothing here feeds back into any loop decision.

    ./core/run.sh python core/final.py microbiome crc_baxter --seeds 1 2 3
Writes projects/<p>/runs/<task>/s<seed>/final/{seed,champion}.json and final_scores.json.
"""
import argparse
import importlib.util
import json
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


def paired_bootstrap(score, y, p_seed, p_champ, groups, task, n_boot=2000, seed=0):
    """Resample SUBJECTS with replacement; recompute champion-minus-seed metric each time."""
    groups = np.asarray([str(g) for g in groups])
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ug, len(ug), replace=True)])
        yy = [y[i] for i in idx]
        try:
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
    ap.add_argument("--time-limit", type=int, default=3600)
    a = ap.parse_args()

    import modal
    cfg = project_cfg(a.project)["tasks"][a.task]
    fn = modal.Function.from_name("jae", f"ev_{cfg.get('size', 'm')}")
    prep = load_prepare(a.project)
    pdir = ROOT / "projects" / a.project
    seed_file = pdir / "seeds" / f"{a.task}.py"
    seed_code = (seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text()
    b = bundle(a.project)

    jobs = {}
    with ThreadPoolExecutor(8) as ex:
        for s in a.seeds:
            run = pdir / "runs" / a.task / f"s{s}"
            for name, code in [("seed", seed_code), ("champion", (run / "solution.py").read_text())]:
                jobs[(s, name)] = ex.submit(fn.remote, a.project, a.task, code, "eval", s, True,
                                            a.time_limit, b)
        res = {k: v.result() for k, v in jobs.items()}

    for s in a.seeds:
        run = pdir / "runs" / a.task / f"s{s}"
        (run / "final").mkdir(exist_ok=True)
        reps = {}
        for name in ("seed", "champion"):
            r = res[(s, name)]
            if r.get("score") is None:
                raise SystemExit(f"s{s} {name} failed on Eval:\n{r.get('log', '')[-2000:]}")
            reps[name] = json.loads(r["report"])
            (run / "final" / f"{name}.json").write_text(r["report"])
        base = json.loads((run / "baseline.json").read_text())
        out = {"project": a.project, "task": a.task, "seed": s,
               "dev_seed": base["dev"], "dev_champion": float((run / "score.txt").read_text()),
               "eval_seed": reps["seed"]["score"], "eval_champion": reps["champion"]["score"]}
        out["dev_gain"] = out["dev_champion"] - out["dev_seed"]
        out["eval_gain"] = out["eval_champion"] - out["eval_seed"]
        rs, rc = reps["seed"], reps["champion"]
        out["eval_bootstrap"] = paired_bootstrap(prep.score, rs["y"], rs["predictions"],
                                                 rc["predictions"], rs["groups"], a.task)
        out["external"] = {}
        for name, ex_c in (rc.get("external") or {}).items():
            ex_s = rs["external"][name]
            e = {"seed": ex_s["score"], "champion": ex_c["score"],
                 "gain": ex_c["score"] - ex_s["score"]}
            e["bootstrap"] = paired_bootstrap(prep.score, ex_s["y"], ex_s["predictions"],
                                              ex_c["predictions"], ex_s["groups"], a.task)
            out["external"][name] = e
        (run / "final_scores.json").write_text(json.dumps(out, indent=2) + "\n")
        ext = "  ".join(f"{k} {v['seed']:.3f}->{v['champion']:.3f}" for k, v in out["external"].items())
        print(f"{a.project}/{a.task} s{s}: Dev {out['dev_seed']:.3f}->{out['dev_champion']:.3f} | "
              f"Eval {out['eval_seed']:.3f}->{out['eval_champion']:.3f} "
              f"(gain {out['eval_gain']:+.3f}, 95% CI {out['eval_bootstrap']['ci95'][0]:+.3f}.."
              f"{out['eval_bootstrap']['ci95'][1]:+.3f}, p={out['eval_bootstrap']['p_one_sided']:.3f})"
              + (f" | external: {ext}" if ext else ""), flush=True)


if __name__ == "__main__":
    main()

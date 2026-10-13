"""
core/search.py -- NON-LLM search baseline with the loop's budget: Optuna TPE over a hand-written,
parameterized pipeline (projects/<p>/baselines/{template,space}.py), 20 Dev evaluations per
task x split, same harness, same Dev CV, same per-candidate time limit as the loop.

Each trial's code is "PARAMS = {...}" + template.py, scored on Dev exactly like a loop candidate.
Champion = best of the seed and the trials (the seed wins ties, as in keep-if-better), then scored
once on Eval with core/final.py --subdir runs_optuna.

    ./core/run.sh python core/search.py microbiome "crc_baxter ibd_pooled" "1 2 3" [--trials 20]
Records: projects/<p>/runs_optuna/<task>/s<seed>/{history.jsonl, solution.py, score.txt,
baseline.json, candidates/round_XX.py}. Resumable: completed trials are replayed into the study.
"""
import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from loop import EPS, NEG_INF, evaluate, project_cfg  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_for(params, template):
    return f"PARAMS = {params!r}\n" + template


def run_one(project, task, seed, trials, time_limit, n_startup):
    pdir = ROOT / "projects" / project
    size = project_cfg(project)["tasks"][task].get("size", "m")
    template = (pdir / "baselines" / "template.py").read_text()
    space = _module(pdir / "baselines" / "space.py")
    run = pdir / "runs_optuna" / task / f"s{seed}"
    (run / "candidates").mkdir(parents=True, exist_ok=True)
    tag = f"{project}/{task} s{seed}"

    loop_run = pdir / "runs" / task / f"s{seed}"
    if not (run / "baseline.json").exists():       # identical seed solution and split as the loop
        shutil.copy(loop_run / "baseline.json", run / "baseline.json")
    best = json.loads((run / "baseline.json").read_text())["dev"]
    if not (run / "solution.py").exists():
        seed_file = pdir / "seeds" / f"{task}.py"
        (run / "solution.py").write_text((seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text())
        (run / "score.txt").write_text(f"{best:.6f}\n")
    best = float((run / "score.txt").read_text())

    sampler = optuna.samplers.TPESampler(seed=1000 + seed, n_startup_trials=n_startup, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    hist_path = run / "history.jsonl"
    history = [json.loads(l) for l in hist_path.read_text().splitlines() if l.strip()] if hist_path.exists() else []
    for h in history:                               # replay finished trials (resume)
        study.enqueue_trial(h["params"])
        t = study.ask()
        space.suggest(t, task)
        if math.isfinite(h["score"]):
            study.tell(t, h["score"])
        else:
            study.tell(t, state=optuna.trial.TrialState.FAIL)
    if history:
        print(f"[search] {tag}: resuming at trial {len(history) + 1} (best {best:.4f})", flush=True)

    for rnd in range(len(history) + 1, trials + 1):
        t = study.ask()
        params = space.suggest(t, task)
        code = code_for(params, template)
        (run / "candidates" / f"round_{rnd:02d}.py").write_text(code)
        t0 = time.time()
        score, log, _ = evaluate(project, task, size, code, seed, time_limit=time_limit)
        kept = score > best + EPS
        if math.isfinite(score):
            study.tell(t, score)
        else:
            study.tell(t, state=optuna.trial.TrialState.FAIL)
        rec = {"round": rnd, "params": params, "score": score,
               "delta": 0.0 if score == NEG_INF else score - best, "kept": kept,
               "seconds": round(time.time() - t0), "strategy": "optuna_tpe",
               "families": [], "rationale": "",
               "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if score == NEG_INF:
            rec["fail_log"] = log[-1500:]
        if kept:
            best = score
            (run / "solution.py").write_text(code)
            (run / "score.txt").write_text(f"{best:.6f}\n")
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        shown = "FAILED" if score == NEG_INF else f"{score:.4f}"
        print(f"[search] {tag} trial {rnd}: {shown} {'KEPT' if kept else ''} (best {best:.4f}) {params}",
              flush=True)
    print(f"[search] {tag}: done, best Dev {best:.4f}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("tasks")
    ap.add_argument("seeds")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-startup", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=1500)
    a = ap.parse_args()
    jobs = [(t, int(s)) for t in a.tasks.split() for s in a.seeds.split()]
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {j: ex.submit(run_one, a.project, j[0], j[1], a.trials, a.time_limit, a.n_startup) for j in jobs}
        for j, f in futs.items():
            try:
                f.result()
            except Exception as e:
                print(f"[search] {a.project}/{j[0]} s{j[1]}: ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

"""
core/audit.py -- transduction audit of the frozen seeds and the loop champions (never feeds back
into any loop decision).

The harness's eval mode featurizes dev-pool and Eval items in ONE featurize call (labels are never
passed, but statistics computed across items -- scaling, prevalence filters, PCA, ComBat, quantile
normalization -- would include the Eval items). For every task x split this script runs the harness
in mode "audit" for the seed and the champion:
  * feat_max_rel_diff: do the dev features change when the Eval items are featurized with them?
    (0 = featurize is per-item, i.e. inductive)
  * score: deployment-style Eval score -- Eval items arrive in 20 chunks, each featurized with the
    dev pool only and predicted alone, vs score_joint (the reported Eval protocol).
  * audit_ref: the same, but each chunk is predicted inside a reference batch (dev pool + chunk),
    so test-batch ranks/scaling in fit_predict use training-time data (deployable variant).
plus a static scan of featurize (and the helpers it calls) for cross-item operations.

    ./core/run.sh python core/audit.py microbiome "crc_baxter ibd_pooled" "1 2 3"
Writes projects/<p>/runs/<task>/s<seed>/final/audit_{seed,champion}[_ref].json and audit.json.
"""
import ast
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bundle import bundle  # noqa: E402
from loop import project_cfg  # noqa: E402

CROSS_ITEM = re.compile(r"\.fit\(|fit_transform|StandardScaler|RobustScaler|QuantileTransformer|PCA\(|"
                        r"ComBat|combat|prevalen|quantile|percentile|nanmean\([^)]*axis\s*=\s*0|"
                        r"mean\([^)]*axis\s*=\s*0|std\([^)]*axis\s*=\s*0|median\([^)]*axis\s*=\s*0|"
                        r"\.mean\(\)|\.std\(\)|rank\(")


BATCH_RELATIVE = re.compile(r"rankdata|\.rank\(|argsort\(\)\.argsort")
HEAVY = {"abnormal_tuab", "autism_abide", "adhd_adhd200", "helsinki"}   # fewer Eval chunks (time limit)


def _reach(funcs, root):
    seen, todo = set(), [root]
    while todo:
        f = todo.pop()
        if f in seen or f not in funcs:
            continue
        seen.add(f)
        for n in ast.walk(funcs[f]):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                if name in funcs:
                    todo.append(name)
    return seen


def static_scan(code):
    """Lines inside featurize (and module functions it reaches) that look like cross-item steps, and
    lines in fit_predict's reach that produce batch-relative outputs (ranks over the test batch)."""
    tree = ast.parse(code)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    lines = code.splitlines()
    rel = []
    for f in sorted(_reach(funcs, "fit_predict")):
        node = funcs[f]
        for i in range(node.lineno - 1, node.end_lineno):
            if BATCH_RELATIVE.search(lines[i]) and not lines[i].strip().startswith("#"):
                rel.append({"function": f, "line": i + 1, "code": lines[i].strip()[:160]})
    if "featurize" not in funcs:
        return {"has_featurize": False, "hits": [], "batch_relative_outputs": rel}
    seen, todo = set(), ["featurize"]
    while todo:
        f = todo.pop()
        if f in seen or f not in funcs:
            continue
        seen.add(f)
        for n in ast.walk(funcs[f]):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                if name in funcs:
                    todo.append(name)
    hits = []
    for f in sorted(seen):
        node = funcs[f]
        for i in range(node.lineno - 1, node.end_lineno):
            if CROSS_ITEM.search(lines[i]) and not lines[i].strip().startswith("#"):
                hits.append({"function": f, "line": i + 1, "code": lines[i].strip()[:160]})
    return {"has_featurize": True, "functions": sorted(seen), "hits": hits, "batch_relative_outputs": rel}


def main():
    import modal
    project, tasks, seeds = sys.argv[1], sys.argv[2].split(), [int(s) for s in sys.argv[3].split()]
    pdir = ROOT / "projects" / project
    b = bundle(project)
    jobs = {}
    with ThreadPoolExecutor(12) as ex:
        for task in tasks:
            fn = modal.Function.from_name("jae", f"ev_{project_cfg(project)['tasks'][task].get('size', 'm')}")
            seed_file = pdir / "seeds" / f"{task}.py"
            seed_code = (seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text()
            for s in seeds:
                run = pdir / "runs" / task / f"s{s}"
                for name, code in [("seed", seed_code), ("champion", (run / "solution.py").read_text())]:
                    for mode, sfx in (("audit", ""), ("audit_ref", "_ref")):
                        if (run / "final" / f"audit_{name}{sfx}.json").exists():
                            continue
                        jobs[(task, s, name, sfx)] = (code, ex.submit(fn.remote, project, task, code,
                                                                      mode + (":10" if task in HEAVY else ""), s,
                                                                      True, 3800, b))
        for (task, s, name, sfx), (code, fut) in jobs.items():
            run = pdir / "runs" / task / f"s{s}"
            r = fut.result()
            out = {"static": static_scan(code)}
            if r.get("score") is None:
                out.update(failed=True, log=r.get("log", "")[-1500:])
            else:
                rep = json.loads(r["report"])
                out.update({k: rep[k] for k in ("score", "score_joint", "feat_max_rel_diff", "n_chunks")})
            (run / "final").mkdir(exist_ok=True)
            (run / "final" / f"audit_{name}{sfx}.json").write_text(json.dumps(out) + "\n")
            msg = "FAILED" if out.get("failed") else (
                f"feat diff {out['feat_max_rel_diff']:.3g} | Eval joint {out['score_joint']:.4f} -> "
                f"{'ref-batch' if sfx else 'chunked'} {out['score']:.4f}")
            print(f"{project}/{task} s{s} {name}{sfx}: {msg} | static hits {len(out['static']['hits'])}",
                  flush=True)
    for task in tasks:                           # compact per-run summary (no predictions)
        seed_file = pdir / "seeds" / f"{task}.py"
        seed_code = (seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text()
        for s in seeds:
            run = pdir / "runs" / task / f"s{s}"
            summ = {}
            for name in ("seed", "champion", "seed_ref", "champion_ref"):
                f = run / "final" / f"audit_{name}.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    summ[name] = {k: v for k, v in d.items() if not k.startswith("predictions")}
                    code = seed_code if name.startswith("seed") else (run / "solution.py").read_text()
                    summ[name]["static"] = static_scan(code)     # always from the current scanner
            (run / "final" / "audit.json").write_text(json.dumps(summ, indent=1) + "\n")


if __name__ == "__main__":
    main()

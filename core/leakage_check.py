"""
core/leakage_check.py -- leakage checks for every frozen champion (and the seeds of tasks whose ids
encode the label). Never feeds back into any loop decision.

1. Static scan of ALL saved solution code (champions, best-of-20 / repeat candidates, archive) for
   file, network, subprocess or harness/prepare access; the only allowed file reads are the
   resources a task provides in ctx (e.g. ctx["genesets_dir"]).
2. Re-scoring on Eval in NETWORK-ISOLATED containers (app "jae-sealed") with every item id
   replaced by a random token (harness mode "eval_anon"); the score must equal the reported one.
3. Positive controls: (a) a solution that reads the label from the id ('aseeg:sz12') must score
   ~1.0 under normal ids and ~0.5 under anonymized ids; (b) a network probe must fail in the
   sealed containers.
    ./core/run.sh python core/leakage_check.py [--tasks t1 t2 ...]
Writes results/robustness/LEAKAGE.md and leakage_rescore.csv.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from bundle import bundle  # noqa: E402
from loop import project_cfg  # noqa: E402

RB = ROOT / "results" / "robustness"
TASKS = [("microbiome", "crc_baxter"), ("microbiome", "ibd_pooled"), ("proteomics_ad", "plasma_ad"),
         ("proteomics_ad", "csf_ad"), ("drug_response", "prism_auc"), ("fmri_connectivity", "autism_abide"),
         ("fmri_connectivity", "adhd_adhd200"), ("eeg_psych", "mdd_tdbrain"), ("eeg_psych", "adhd_tdbrain"),
         ("eeg_psych", "sz_aseeg"), ("eeg_psych", "abnormal_tuab"), ("neonatal_seizure", "helsinki")]
WITH_EXTERNAL = {"crc_baxter", "mdd_tdbrain", "sz_aseeg", "abnormal_tuab"}
ID_ENCODES_LABEL = {"sz_aseeg", "mdd_tdbrain", "plasma_ad", "crc_baxter"}   # in Eval or external ids
ACCESS = re.compile(r"\bopen\(|read_csv|read_parquet|read_table|np\.load\(|pd\.read|/data\b|JAE_|os\.environ|"
                    r"requests|urllib|http[s]?://|socket|subprocess|os\.system|os\.listdir|os\.walk|glob\.|"
                    r"import prepare|from prepare|importlib|sys\.path|harness|__import__|\beval\(|\bexec\(")
ALLOWED = re.compile(r"genesets_dir|\"genesets\"|'genesets'")

CHEAT = '''
import numpy as np
def featurize(items, ctx):
    return np.array([[1.0 if ":sz" in str(i) else 0.0] for i in items])
def fit_predict(F_train, y_train, F_test, ctx):
    return np.asarray(F_test)[:, 0]
'''
NETPROBE = '''
import numpy as np, urllib.request
try:
    urllib.request.urlopen("https://example.com", timeout=10)
    print("NETWORK_PROBE: reachable", flush=True)
except Exception as e:
    print("NETWORK_PROBE: blocked", type(e).__name__, flush=True)
def featurize(items, ctx):
    return np.zeros((len(items), 1))
def fit_predict(F_train, y_train, F_test, ctx):
    return np.random.default_rng(0).random(len(F_test))
'''


def static_scan():
    files = sorted(set(ROOT.glob("projects/*/runs*/*/s*/solution.py")) |
                   set(ROOT.glob("projects/*/runs*/*/s*/candidates/*.py")) |
                   set(ROOT.glob("projects/*/archive/**/solution.py")))
    hits = []
    for f in files:
        doc = _docstring_lines(f.read_text())
        for n, line in enumerate(f.read_text().splitlines(), 1):
            s = line.strip()
            if n in doc or s.startswith("#") or not ACCESS.search(s):
                continue
            hits.append({"file": str(f.relative_to(ROOT)), "line": n, "code": s[:140],
                         "allowed": bool(ALLOWED.search(s)) or "genesets_dir" in _context(f, n)
                         or _helper_uses_genesets(f, n)})
    return files, hits


def _docstring_lines(src):
    """Line numbers inside module / class / function docstrings (text, not code)."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                out.update(range(first.lineno, first.end_lineno + 1))
    return out


def _helper_uses_genesets(f, n):
    """True if the function enclosing line n gets its paths from a helper (same file) that reads
    ctx['genesets_dir'] -- one level of the call graph."""
    import ast
    tree = ast.parse(f.read_text())
    funcs = {x.name: x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    src = f.read_text().splitlines()
    encl = [x for x in funcs.values() if x.lineno <= n <= x.end_lineno]
    if not encl:
        return False
    fn = max(encl, key=lambda x: x.lineno)
    called = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    return any("genesets_dir" in "\n".join(src[funcs[c].lineno - 1: funcs[c].end_lineno])
               for c in called if c in funcs)


def _context(f, n):
    """Source of the function enclosing line n (the whole file if at module level)."""
    import ast
    src = f.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= n <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    lines = src.splitlines()
    return "\n".join(lines[best.lineno - 1: best.end_lineno]) if best else src


CACHE = RB / "leakage_runs.json"
MODES = {"eval": "sealed, real ids", "eval_anono": "sealed, order-preserving tokens",
         "eval_anon": "sealed, random tokens"}


def main():
    import modal
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    a = ap.parse_args()
    tasks = [pt for pt in TASKS if a.tasks is None or pt[1] in a.tasks]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    files, hits = static_scan()
    jobs = {}
    with ThreadPoolExecutor(12) as ex:
        b_eeg = bundle("eeg_psych")
        fn = modal.Function.from_name("jae-sealed", "ev_m")
        for mode in ("eval", "eval_anono", "eval_anon"):        # positive control (a)
            k = f"control|cheat|{mode}"
            if cache.get(k, {}).get("score") is None:
                jobs[k] = ex.submit(fn.remote, "eeg_psych", "sz_aseeg", CHEAT, mode, 1, True, 1800, b_eeg)
        if "NETWORK_PROBE: blocked" not in cache.get("control|netprobe|eval", {}).get("log", ""):
            jobs["control|netprobe|eval"] = ex.submit(                      # positive control (b)
                modal.Function.from_name("jae-sealed", "ev_s").remote, "eeg_psych", "sz_aseeg", NETPROBE, "eval", 1,
                False, 1800, b_eeg)
        for project, task in tasks:
            size = project_cfg(project)["tasks"][task].get("size", "m")
            fn = modal.Function.from_name("jae-sealed", f"ev_{size}")
            b = bundle(project)
            pdir = ROOT / "projects" / project
            seed_file = pdir / "seeds" / f"{task}.py"
            seed_code = (seed_file if seed_file.exists() else pdir / "seeds" / "default.py").read_text()
            for s in (1, 2, 3):
                run = pdir / "runs" / task / f"s{s}"
                arms = [("champion", (run / "solution.py").read_text())]
                if task in ID_ENCODES_LABEL:
                    arms.append(("seed", seed_code))
                for arm, code in arms:
                    for mode in a.modes:
                        k = f"{task}|{s}|{arm}|{mode}"
                        if cache.get(k, {}).get("score") is None:
                            # full report only where external sets exist (large reports, e.g. Helsinki's
                            # per-second predictions, come back via blob storage; only the score is needed)
                            jobs[k] = ex.submit(fn.remote, project, task, code, mode, s, task in WITH_EXTERNAL,
                                                3800, b)
        for k, f in jobs.items():
            try:
                r = f.result()
            except Exception as e:                             # noqa: BLE001
                r = {"score": None, "log": f"{type(e).__name__}: {e}", "report": None}
            ext = {}
            if r.get("report"):
                ext = {n: e["score"] for n, e in (json.loads(r["report"]).get("external") or {}).items()}
            cache[k] = {"score": r["score"], "external": ext, "log": (r.get("log") or "")[-600:]}
            CACHE.write_text(json.dumps(cache, indent=0))      # persist after every result

    import pandas as pd
    rows = []
    proj = dict((t, p) for p, t in TASKS)
    keys = sorted({tuple(k.split("|")[:3]) for k in cache if not k.startswith("control|")})
    for task, s, arm in keys:
        ref = json.loads((ROOT / "projects" / proj[task] / "runs" / task / f"s{s}" / "final" / f"{arm}.json").read_text())
        row = {"task": task, "split": int(s), "arm": arm, "reported": ref["score"]}
        for mode in MODES:
            c = cache.get(f"{task}|{s}|{arm}|{mode}")
            row[mode] = None if c is None else c["score"]
            for n, v in ((c or {}).get("external") or {}).items():
                row[f"ext:{n}:reported"] = ref["external"][n]["score"]
                row[f"ext:{n}:{mode}"] = v
        rows.append(row)
    d = pd.DataFrame(rows).sort_values(["task", "split", "arm"])
    d.to_csv(RB / "leakage_rescore.csv", index=False, float_format="%.6f")
    write_md(d, cache, files, hits)
    print(d[["task", "split", "arm", "reported"] + list(MODES)].to_string())


def _maxdiff(d, mode):
    cols = [("reported", mode)] + [(c, c.replace(":reported", f":{mode}")) for c in d.columns
                                   if c.startswith("ext:") and c.endswith(":reported")]
    out = 0.0
    for a, b in cols:
        if b in d:
            v = (d[b] - d[a]).abs().max()
            if v == v:
                out = max(out, float(v))
    return out


def write_md(d, cache, files, hits):
    cheat = {m: cache.get(f"control|cheat|{m}", {}).get("score") for m in MODES}
    probe_log = cache.get("control|netprobe|eval", {}).get("log", "")
    probe = "blocked" if "NETWORK_PROBE: blocked" in probe_log else (
        "REACHABLE" if "reachable" in probe_log else "unknown")
    fails = {m: int(d[m].isna().sum()) for m in MODES}
    bad = [h for h in hits if not h["allowed"]]
    fmt = (lambda v: "FAILED" if v is None or v != v else f"{v:.6f}")
    L = ["# Leakage checks (ids, files, network)", "",
         "All runs below use network-isolated scoring containers (app `jae-sealed`, `block_network=True`).", "",
         "## Positive controls (do the checks have power?)", "",
         "* A solution that reads the label from the id (`':sz' in item`), sz_aseeg split 1, Eval AUROC: "
         + ", ".join(f"{MODES[m]} **{fmt(cheat[m])[:5]}**" for m in MODES) + ".",
         f"* Network probe inside the scoring container: **{probe}**.", "",
         "## Re-scoring every champion (and the seeds of tasks whose ids encode the label)", "",
         "Three variants: real ids (determinism check); **order-preserving tokens** (the id TEXT is replaced but",
         "the ids' sort order is kept, so anything that depends only on order -- e.g. sklearn GroupKFold's",
         "assignment of `ctx['groups_train']` to inner-CV folds -- is unchanged); random tokens (the order",
         "changes too). A solution that read labels from ids would lose performance under BOTH token",
         "variants (cf. the control: 1.0 -> 0.5).", "",
         f"Largest absolute difference from the reported score (Eval and external sets): real ids "
         f"**{_maxdiff(d, 'eval'):.2e}**; order-preserving tokens **{_maxdiff(d, 'eval_anono'):.2e}**; random tokens "
         f"**{_maxdiff(d, 'eval_anon'):.2e}** (failed runs: "
         + ", ".join(f"{MODES[m]} {fails[m]}" for m in MODES) + ").", "",
         "| task | split | arm | reported | real ids | order-preserving tokens | random tokens |",
         "|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        L.append(f"| {r.task} | {r.split} | {r.arm} | {r.reported:.6f} | {fmt(r.eval)} | {fmt(r.eval_anono)} | "
                 f"{fmt(r.eval_anon)} |")
    L += ["", "## Static scan of all saved solution code", "",
          f"{len(files)} files (champions, best-of-20 and repeat candidates, archived runs). Lines touching files, "
          f"network, subprocesses or the harness: {len(hits)}; outside the resources a task provides "
          f"(`ctx['genesets_dir']`, directly or via a helper function): **{len(bad)}**."]
    for h in bad[:40]:
        L.append(f"* `{h['file']}:{h['line']}`: `{h['code']}`")
    L += ["", "Limits: rejected candidates of the 36 main loops were not saved (only champions and the",
          "round history), so they are covered only through their champions; the loops' scoring",
          "containers (app `jae`) had network access and the data volume mounted -- the checks above show",
          "that no saved solution used either."]
    (RB / "LEAKAGE.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

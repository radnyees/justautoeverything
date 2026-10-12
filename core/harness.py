"""
core/harness.py -- the FIXED evaluation harness shared by every project.

The LLM edits only a project's `solution.py`; this file and the project's `prepare.py` are
never edited by the loop. The harness owns the data split, the cross-validation, the metric
and the label handling, so a candidate can only influence its predictions -- it never
receives the labels of the items it is scored on.

Protocol (identical for every project/task):
  * A task's dataset is split once per SEED into a DEV POOL (75% of groups) and a held-out
    EVAL set (25% of groups), grouped by subject (no subject on both sides) and stratified
    by the task's strata (label, and site/study where relevant).
  * mode "dev"  -- K-fold grouped, stratified CV inside the dev pool. The loop optimizes this
                   out-of-fold score. Eval items are never featurized, fitted or scored.
  * mode "eval" -- fit once on the whole dev pool, score the held-out Eval set (and any
                   external datasets). Run ONCE per champion, after the loop has finished.

Solution contract (projects/<p>/solution files):
    def featurize(items, ctx) -> features aligned with items      # optional, label-free
    def fit_predict(F_train, y_train, F_test, ctx) -> predictions # one prediction per test item
`featurize` is called once on every item a mode needs (dev pool, or dev pool + Eval) and must
not use labels; `fit_predict` sees training labels only.

Run (inside the Modal container, cwd = repo root):
    JAE_PROJECT=eeg_psych JAE_TASK=mdd_mumtaz JAE_SEED=1 JAE_MODE=dev python core/harness.py path/to/solution.py
Prints exactly one line `SCORE: <float>` (plus diagnostics). JAE_REPORT=1 also writes
predictions to $JAE_REPORT_PATH as JSON.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_FRAC = 0.25


# ---------------------------------------------------------------- indexing helpers
def take(obj, idx):
    """Subset arrays, DataFrames, lists or dicts-of-those by integer positions."""
    idx = np.asarray(idx, dtype=int)
    if obj is None:
        return None
    if hasattr(obj, "iloc"):
        return obj.iloc[idx]
    if isinstance(obj, np.ndarray):
        return obj[idx]
    if isinstance(obj, dict):
        return {k: take(v, idx) for k, v in obj.items()}
    return [obj[i] for i in idx]


def concat(parts):
    """Concatenate featurize/predict outputs produced per chunk."""
    if hasattr(parts[0], "iloc"):
        import pandas as pd
        return pd.concat(parts, axis=0)
    if isinstance(parts[0], np.ndarray):
        return np.concatenate(parts, axis=0)
    out = []
    for p in parts:
        out.extend(p)
    return out


# ---------------------------------------------------------------- splitting
def grouped_stratified_split(groups, strata, frac, seed):
    """Pick ~frac of the GROUPS (subjects) for the held-out side, stratified by the
    group's stratum (a group's stratum = the most common stratum of its items)."""
    groups = np.asarray(groups).astype(str)
    strata = np.asarray(strata).astype(str)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    g_stratum = {}
    for g in ug:
        vals, cnt = np.unique(strata[groups == g], return_counts=True)
        g_stratum[g] = vals[np.argmax(cnt)]
    held = []
    for s in sorted(set(g_stratum.values())):
        gs = np.array(sorted(g for g in ug if g_stratum[g] == s))
        rng.shuffle(gs)
        k = int(round(frac * len(gs)))
        if len(gs) >= 2:
            k = min(max(k, 1), len(gs) - 1)
        held.extend(gs[:k])
    held = set(held)
    mask = np.array([g in held for g in groups])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def grouped_stratified_kfold(groups, strata, k, seed):
    """K grouped folds with each group's stratum spread round-robin across folds."""
    groups = np.asarray(groups).astype(str)
    strata = np.asarray(strata).astype(str)
    rng = np.random.default_rng(seed + 1000)
    ug = np.unique(groups)
    g_stratum = {}
    for g in ug:
        vals, cnt = np.unique(strata[groups == g], return_counts=True)
        g_stratum[g] = vals[np.argmax(cnt)]
    fold_of = {}
    for s in sorted(set(g_stratum.values())):
        gs = np.array(sorted(g for g in ug if g_stratum[g] == s))
        rng.shuffle(gs)
        offset = rng.integers(k)
        for i, g in enumerate(gs):
            fold_of[g] = (i + offset) % k
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


# ---------------------------------------------------------------- running a solution
def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fit_predict"):
        raise SystemExit("solution must define fit_predict(F_train, y_train, F_test, ctx)")
    return mod


def run(solution_path):
    project = os.environ["JAE_PROJECT"]
    task = os.environ["JAE_TASK"]
    seed = int(os.environ.get("JAE_SEED", "1"))
    mode = os.environ.get("JAE_MODE", "dev")
    sys.path.insert(0, str(ROOT / "projects" / project))
    sys.path.insert(0, str(ROOT))
    prepare = importlib.import_module("prepare")
    cfg = prepare.TASKS[task]
    t0 = time.time()
    ds = prepare.load(task)                      # dict: items, y, groups, strata, meta?, ctx?, external?
    sol = load_solution(solution_path)
    n = len(ds["groups"])
    dev_idx, eval_idx = grouped_stratified_split(ds["groups"], ds["strata"], EVAL_FRAC, seed)
    ctx_base = dict(ds.get("ctx", {}), task=task, seed=seed)
    print(f"[harness] {project}/{task} seed={seed} mode={mode} items={n} "
          f"dev_pool={len(dev_idx)} eval={len(eval_idx)} load={time.time() - t0:.0f}s", flush=True)

    def featurize(idx, items=None, meta=None):
        items = take(ds["items"], idx) if items is None else items
        meta = take(ds.get("meta"), idx) if meta is None else meta
        if hasattr(sol, "featurize"):
            return sol.featurize(items, dict(ctx_base, meta=meta))
        return items

    report = {"project": project, "task": task, "seed": seed, "mode": mode}
    if mode == "dev":
        F = featurize(dev_idx)
        groups, strata = take(ds["groups"], dev_idx), take(ds["strata"], dev_idx)
        y = take(ds["y"], dev_idx)
        meta = take(ds.get("meta"), dev_idx)
        folds = grouped_stratified_kfold(groups, strata, cfg.get("cv_folds", 5), seed)
        oof = [None] * len(dev_idx)
        for tr, te in folds:
            ctx = dict(ctx_base, meta_train=take(meta, tr), meta_test=take(meta, te),
                       groups_train=take(groups, tr))
            pred = sol.fit_predict(take(F, tr), take(y, tr), take(F, te), ctx)
            for j, p in zip(te, list(pred)):
                oof[j] = p
        score = prepare.score(y, oof, task)
        report["score"] = score
        report["predictions"] = _jsonable(oof)
        report["y"] = _jsonable(y)
        report["groups"] = _jsonable(groups)
    elif mode == "eval":
        both = np.concatenate([dev_idx, eval_idx])
        F = featurize(both)
        F_dev, F_ev = take(F, np.arange(len(dev_idx))), take(F, np.arange(len(dev_idx), len(both)))
        ctx = dict(ctx_base, meta_train=take(ds.get("meta"), dev_idx),
                   meta_test=take(ds.get("meta"), eval_idx), groups_train=take(ds["groups"], dev_idx))
        pred = sol.fit_predict(F_dev, take(ds["y"], dev_idx), F_ev, ctx)
        y_ev = take(ds["y"], eval_idx)
        score = prepare.score(y_ev, list(pred), task)
        report.update(score=score, predictions=_jsonable(list(pred)), y=_jsonable(y_ev),
                      groups=_jsonable(take(ds["groups"], eval_idx)))
        ext = {}
        for name, ex in (ds.get("external") or {}).items():   # other datasets, scored once
            Fx = sol.featurize(ex["items"], dict(ctx_base, meta=ex.get("meta"))) \
                if hasattr(sol, "featurize") else ex["items"]
            px = sol.fit_predict(F_dev, take(ds["y"], dev_idx), Fx,
                                 dict(ctx, meta_test=ex.get("meta")))
            ext[name] = {"score": prepare.score(ex["y"], list(px), task),
                         "predictions": _jsonable(list(px)), "y": _jsonable(ex["y"]),
                         "groups": _jsonable(ex["groups"])}
            print(f"[harness] external {name}: SCORE {ext[name]['score']:.6f}", flush=True)
        report["external"] = ext
    else:
        raise SystemExit(f"unknown JAE_MODE {mode}")

    print(f"[harness] done in {time.time() - t0:.0f}s", flush=True)
    if os.environ.get("JAE_REPORT"):
        Path(os.environ.get("JAE_REPORT_PATH", "report.json")).write_text(json.dumps(report))
    print(f"SCORE: {score:.6f}", flush=True)


def _jsonable(x):
    if x is None:
        return None
    if hasattr(x, "to_numpy"):                     # pandas DataFrame / Series
        return x.to_numpy().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


if __name__ == "__main__":
    run(sys.argv[1])

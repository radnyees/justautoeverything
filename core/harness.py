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


# ---------------------------------------------------------------- id anonymization (audit)
def anonymize(ds, seed, ordered=False):
    """Replace every item id the solution can see (list items, DataFrame indexes, meta indexes,
    the group ids passed as ctx["groups_train"], external items) by random tokens; callables in
    ctx get the real ids back, so data loading still works. Splits, labels and reports keep the
    real ids. Used to verify that no solution reads labels from ids such as 'aseeg:sz12'."""
    rng = np.random.default_rng(10_000 + seed)
    real = []

    def ids_of(items):
        return [str(i) for i in (items.index if hasattr(items, "iloc") else items)]
    real += ids_of(ds["items"])
    for ex in (ds.get("external") or {}).values():
        real += ids_of(ex["items"])
    real += [str(g) for g in ds["groups"]]
    uniq = sorted(set(real))
    # ordered=True: token rank = rank of the real id, so anything that depends only on the ORDER of ids
    # (e.g. sklearn GroupKFold's assignment of groups to folds) is unchanged, while the id TEXT is gone
    nums = np.arange(len(uniq)) if ordered else rng.permutation(len(uniq))
    tok = {r: f"x{n:06d}" for r, n in zip(uniq, nums)}
    inv = {t: r for r, t in tok.items()}

    def swap_items(items):
        if hasattr(items, "iloc"):
            out = items.copy()
            out.index = [tok[str(i)] for i in items.index]
            return out
        return [tok[str(i)] for i in items]

    def swap_meta(meta):
        if meta is None or not hasattr(meta, "index"):
            return meta
        if all(str(i) in tok for i in meta.index):
            out = meta.copy()
            out.index = [tok[str(i)] for i in meta.index]
            return out
        return meta

    def untok(x):
        if isinstance(x, str):
            return inv.get(x, x)
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) and all(isinstance(v, str) for v in x):
            return [inv.get(v, v) for v in x]
        return x

    def wrap(f):
        def g(*a, **k):
            return f(*[untok(v) for v in a], **{kk: untok(v) for kk, v in k.items()})
        return g

    ds = dict(ds)
    ds["items"] = swap_items(ds["items"])
    ds["meta"] = swap_meta(ds.get("meta"))
    ds["groups_ctx"] = np.array([tok[str(g)] for g in ds["groups"]])
    ds["ctx"] = {k: (wrap(v) if callable(v) else v) for k, v in (ds.get("ctx") or {}).items()}
    ds["external"] = {name: dict(ex, items=swap_items(ex["items"]), meta=swap_meta(ex.get("meta")))
                      for name, ex in (ds.get("external") or {}).items()}
    return ds


# ---------------------------------------------------------------- running a solution
def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fit_predict"):
        raise SystemExit("solution must define fit_predict(F_train, y_train, F_test, ctx)")
    return mod


def _score(prepare, y, pred, task, sites):
    """prepare.score(y, pred, task[, sites]) -- sites only for tasks scored within site."""
    import inspect
    if sites is not None and len(inspect.signature(prepare.score).parameters) >= 4:
        return prepare.score(y, pred, task, sites)
    return prepare.score(y, pred, task)


def run(solution_path):
    project = os.environ["JAE_PROJECT"]
    task = os.environ["JAE_TASK"]
    seed = int(os.environ.get("JAE_SEED", "1"))
    mode = os.environ.get("JAE_MODE", "dev")
    if ":" in mode:                              # e.g. "audit:10" = audit with 10 Eval chunks
        mode, os.environ["JAE_AUDIT_CHUNKS"] = mode.split(":", 1)
    anon = None                                  # "eval_anon": random tokens; "eval_anono": tokens that
    for sfx, kind in (("_anono", "ordered"), ("_anon", "random")):   # keep the ids' sort order
        if mode.endswith(sfx):
            anon, mode = kind, mode[: -len(sfx)]
            break
    sys.path.insert(0, str(ROOT / "projects" / project))
    sys.path.insert(0, str(ROOT))
    prepare = importlib.import_module("prepare")
    cfg = prepare.TASKS[task]
    t0 = time.time()
    ds = prepare.load(task)                      # dict: items, y, groups, strata, meta?, ctx?, external?
    if anon:
        ds = anonymize(ds, seed, ordered=(anon == "ordered"))
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
        sites = take(ds.get("sites"), dev_idx)
        score = _score(prepare, y, oof, task, sites)
        report["score"] = score
        report["sites"] = _jsonable(sites)
        report["predictions"] = _jsonable(oof)
        report["y"] = _jsonable(y)
        report["groups"] = _jsonable(groups)
    elif mode == "eval":
        both = np.concatenate([dev_idx, eval_idx])
        F = featurize(both)
        F_dev, F_ev = take(F, np.arange(len(dev_idx))), take(F, np.arange(len(dev_idx), len(both)))
        ctx = dict(ctx_base, meta_train=take(ds.get("meta"), dev_idx),
                   meta_test=take(ds.get("meta"), eval_idx),
                   groups_train=take(ds.get("groups_ctx", ds["groups"]), dev_idx))
        pred = sol.fit_predict(F_dev, take(ds["y"], dev_idx), F_ev, ctx)
        y_ev = take(ds["y"], eval_idx)
        sites = take(ds.get("sites"), eval_idx)
        score = _score(prepare, y_ev, list(pred), task, sites)
        report.update(score=score, predictions=_jsonable(list(pred)), y=_jsonable(y_ev),
                      groups=_jsonable(take(ds["groups"], eval_idx)), sites=_jsonable(sites))
        ext = {}
        for name, ex in (ds.get("external") or {}).items():   # other datasets, scored once
            Fx = sol.featurize(ex["items"], dict(ctx_base, meta=ex.get("meta"))) \
                if hasattr(sol, "featurize") else ex["items"]
            px = sol.fit_predict(F_dev, take(ds["y"], dev_idx), Fx,
                                 dict(ctx, meta_test=ex.get("meta")))
            ext[name] = {"score": _score(prepare, ex["y"], list(px), task, ex.get("sites")),
                         "predictions": _jsonable(list(px)), "y": _jsonable(ex["y"]),
                         "groups": _jsonable(ex["groups"]), "sites": _jsonable(ex.get("sites"))}
            print(f"[harness] external {name}: SCORE {ext[name]['score']:.6f}", flush=True)
        report["external"] = ext
    elif mode in ("audit", "audit_ref"):
        # Transduction audit (never used for selection). (1) featurize: do the dev-pool features
        # change when the Eval items are featurized together with them (as in mode eval)?
        # (2) deployment-style Eval: dev features computed WITHOUT any Eval item; Eval items arrive
        # in small chunks, each featurized with the dev pool only and predicted on its own, so no
        # statistic can be shared across Eval items. Equal to mode eval for inductive pipelines.
        # audit_ref: each chunk is predicted inside a reference batch = dev pool + chunk (only the
        # chunk's outputs are kept), so batch-relative steps in fit_predict (test-set ranks/scaling)
        # use data available at training time -- a deployable version of such pipelines.
        both = np.concatenate([dev_idx, eval_idx])
        nd = len(dev_idx)
        F_all = featurize(both)
        F_dev_joint, F_ev_joint = take(F_all, np.arange(nd)), take(F_all, np.arange(nd, len(both)))
        F_dev = featurize(dev_idx)
        feat_diff = _max_rel_diff(F_dev_joint, F_dev)
        y_dev, y_ev = take(ds["y"], dev_idx), take(ds["y"], eval_idx)
        ctx = dict(ctx_base, meta_train=take(ds.get("meta"), dev_idx),
                   meta_test=take(ds.get("meta"), eval_idx), groups_train=take(ds["groups"], dev_idx))
        sites = take(ds.get("sites"), eval_idx)
        pred_joint = list(sol.fit_predict(F_dev_joint, y_dev, F_ev_joint, ctx))
        n_chunks = min(len(eval_idx), int(os.environ.get("JAE_AUDIT_CHUNKS", "20")))
        order = np.random.default_rng(seed).permutation(len(eval_idx))
        pred_ind = [None] * len(eval_idx)
        for c, pos in enumerate(np.array_split(order, n_chunks)):
            pos = np.sort(pos)
            if feat_diff > 1e-6:                # batch-dependent featurize: dev pool + this chunk only
                Fx = featurize(np.concatenate([dev_idx, eval_idx[pos]]))
                F_tr, Fc = take(Fx, np.arange(nd)), take(Fx, np.arange(nd, nd + len(pos)))
            else:                               # featurize is per-item: featurize the chunk alone
                F_tr, Fc = F_dev, featurize(eval_idx[pos])
            if mode == "audit_ref":
                ref_idx = np.concatenate([dev_idx, eval_idx[pos]])
                cctx = dict(ctx, meta_test=take(ds.get("meta"), ref_idx))
                out = list(sol.fit_predict(F_tr, y_dev, _concat(F_tr, Fc), cctx))[nd:]
            else:
                cctx = dict(ctx, meta_test=take(ds.get("meta"), eval_idx[pos]))
                out = list(sol.fit_predict(F_tr, y_dev, Fc, cctx))
            for j, p in zip(pos, out):
                pred_ind[j] = p
            print(f"[harness] audit chunk {c + 1}/{n_chunks}", flush=True)
        s_joint = _score(prepare, y_ev, pred_joint, task, sites)
        score = _score(prepare, y_ev, pred_ind, task, sites)
        print(f"[harness] audit: featurize max rel diff {feat_diff:.3g}; Eval joint {s_joint:.6f} "
              f"-> chunked inductive {score:.6f}", flush=True)
        report.update(score=score, score_joint=s_joint, feat_max_rel_diff=feat_diff, n_chunks=n_chunks,
                      predictions=_jsonable(pred_ind), predictions_joint=_jsonable(pred_joint),
                      y=_jsonable(y_ev), groups=_jsonable(take(ds["groups"], eval_idx)),
                      sites=_jsonable(sites))
    else:
        raise SystemExit(f"unknown JAE_MODE {mode}")

    print(f"[harness] done in {time.time() - t0:.0f}s", flush=True)
    if os.environ.get("JAE_REPORT"):
        Path(os.environ.get("JAE_REPORT_PATH", "report.json")).write_text(json.dumps(report))
    print(f"SCORE: {score:.6f}", flush=True)


def _concat(a, b):
    """Row-concatenate two feature batches of the same type (array, DataFrame, list, dict)."""
    if hasattr(a, "iloc"):
        import pandas as pd
        return pd.concat([a, b])
    if isinstance(a, np.ndarray):
        return np.concatenate([a, b])
    if isinstance(a, dict):
        return {k: _concat(a[k], b[k]) for k in a}
    return list(a) + list(b)


def _max_rel_diff(a, b):
    """Largest |a - b| / (|b| scale) over all feature values (lists of per-item arrays allowed);
    inf if the shapes differ."""
    def flat(x):
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        if isinstance(x, dict):
            return [u for k in sorted(x) for u in flat(x[k])]
        if isinstance(x, (list, tuple)):
            return [u for v in x for u in flat(v)]
        return [np.asarray(x, dtype=float).ravel()]
    try:
        fa, fb = flat(a), flat(b)
    except (TypeError, ValueError):            # non-numeric features: treat as batch-dependent
        return float("inf")
    if len(fa) != len(fb) or any(u.shape != v.shape for u, v in zip(fa, fb)):
        return float("inf")
    ua, ub = np.concatenate(fa), np.concatenate(fb)
    ok = np.isfinite(ua) & np.isfinite(ub)
    if (np.isfinite(ua) != np.isfinite(ub)).any():
        return float("inf")
    scale = np.abs(ub[ok]).mean() + 1e-12 if ok.any() else 1.0
    return float(np.abs(ua[ok] - ub[ok]).max() / scale) if ok.any() else 0.0


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

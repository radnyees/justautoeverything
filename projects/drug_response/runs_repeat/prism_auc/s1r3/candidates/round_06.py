"""Top-5,000 variable genes, standardized, multi-output ridge (alpha=5000), written in kernel (dual) form
so each drug can get extra, less-penalized blocks: (1) its own annotated target genes and (2) the targets'
top STRING first-degree neighbours. Global block weights are chosen by grouped inner CV
(weights 0 == original seed model)."""
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
NB_K = 25
GRID = [(0.0, 0.0)] + [(s, r) for s in (100.0, 400.0, 1500.0) for r in (0.0, 0.5, 1.0)]

_STRING_CACHE = {}


def _drug_targets(ctx, cols, gene_set):
    d = ctx["drugs"].copy()
    if "broad_id" in d.columns and set(cols).issubset(set(d["broad_id"])):
        d = d.drop_duplicates("broad_id").set_index("broad_id").reindex(cols)
    else:
        d = d.reset_index(drop=True)
    out = []
    for i in range(len(cols)):
        t = d["target"].iloc[i] if i < len(d) else None
        if not isinstance(t, str):
            out.append([])
            continue
        gs = [g.strip().upper() for g in re.split(r"[,;|/]", t)]
        gs = sorted({g for g in gs if g in gene_set})
        out.append(gs)
    return out


def _string_paths(ctx):
    gdir = ctx.get("genesets_dir")
    gs = ctx.get("genesets")
    cands = []
    vals = []
    if isinstance(gs, dict):
        for k, v in gs.items():
            if isinstance(v, str):
                vals.append((str(k), v))
    elif isinstance(gs, (list, tuple)):
        vals = [(str(v), v) for v in gs if isinstance(v, str)]
    elif isinstance(gs, str):
        vals = [(gs, gs)]
    for k, v in vals:
        if any(w in (k + v).lower() for w in ("string", "ppi")):
            p = v if (os.path.isabs(v) or not gdir) else os.path.join(gdir, v)
            cands.append(p)
    if gdir and os.path.isdir(gdir):
        for root, _, files in os.walk(gdir):
            for f in files:
                fl = f.lower()
                if ("string" in fl or "ppi" in fl) and not fl.endswith(".gmt"):
                    cands.append(os.path.join(root, f))
    seen, out = set(), []
    for p in cands:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            out.append(p)
    return out


def _load_string(ctx):
    if "adj" in _STRING_CACHE:
        return _STRING_CACHE["adj"]
    adj = {}
    for p in _string_paths(ctx):
        df = None
        for sep in ("\t", ",", r"\s+"):
            try:
                tmp = pd.read_csv(p, sep=sep, engine="c" if sep != r"\s+" else "python")
            except Exception:
                continue
            if tmp.shape[1] >= 2:
                df = tmp
                break
        if df is None:
            continue
        obj = [c for c in df.columns if df[c].dtype == object]
        if len(obj) < 2:
            continue
        num = [c for c in df.columns if c not in obj and np.issubdtype(df[c].dtype, np.number)]
        sc = [c for c in num if "score" in str(c).lower()]
        scol = sc[0] if sc else (num[-1] if num else None)
        a = df[obj[0]].astype(str).str.upper().values
        b = df[obj[1]].astype(str).str.upper().values
        w = df[scol].astype(float).values if scol is not None else np.ones(len(df))
        for x, y, s in zip(a, b, w):
            if x == y:
                continue
            dx = adj.setdefault(x, {})
            dx[y] = max(dx.get(y, 0.0), s)
            dy = adj.setdefault(y, {})
            dy[x] = max(dy.get(x, 0.0), s)
        if adj:
            break
    _STRING_CACHE["adj"] = adj
    return adj


def _drug_neighbours(targets, adj, gene_set):
    out = []
    for t in targets:
        if not t or not adj:
            out.append([])
            continue
        tset = set(t)
        agg = {}
        for g in t:
            for nb, s in adj.get(g, {}).items():
                if nb in tset or nb not in gene_set:
                    continue
                agg[nb] = agg.get(nb, 0.0) + s
        top = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_K]
        out.append(sorted(g for g, _ in top))
    return out


def _predict_grid(F_train, Y, F_test, targets, nbrs, grid):
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    K = Xtr @ Xtr.T
    Kx = Xte @ Xtr.T
    n = K.shape[0]
    ym = Y.mean(0)
    Yc = Y - ym
    base = Kx @ np.linalg.solve(K + ALPHA * np.eye(n), Yc) + ym
    preds = {c: base.copy() for c in grid}
    allg = sorted({g for t in targets for g in t} | {g for t in nbrs for g in t})
    allg = [g for g in allg if var.get(g, 0) > 1e-8]
    if not allg:
        return preds
    gset = set(allg)
    zmu, zsd = F_train[allg].mean(), F_train[allg].std(ddof=0).replace(0, 1.0)
    Ztr = (F_train[allg] - zmu) / zsd
    Zte = (F_test[allg] - zmu) / zsd
    active = [c for c in grid if c[0] > 0]
    if not active:
        return preds
    I = ALPHA * np.eye(n)
    for j in range(len(targets)):
        t = [g for g in targets[j] if g in gset]
        if not t:
            continue
        A = Ztr[t].values
        B = Zte[t].values
        KT = (A @ A.T) / len(t)
        KxT = (B @ A.T) / len(t)
        nb = [g for g in nbrs[j] if g in gset]
        if nb:
            An = Ztr[nb].values
            Bn = Zte[nb].values
            KN = (An @ An.T) / len(nb)
            KxN = (Bn @ An.T) / len(nb)
        for (s, r) in active:
            if r > 0 and nb:
                Kd = K + s * KT + s * r * KN
                Kxd = Kx + s * KxT + s * r * KxN
            else:
                Kd = K + s * KT
                Kxd = Kx + s * KxT
            preds[(s, r)][:, j] = Kxd @ np.linalg.solve(Kd + I, Yc[:, j]) + ym[j]
    return preds


def _mean_pearson(P, Yobs):
    rs = []
    for j in range(Yobs.shape[1]):
        m = ~np.isnan(Yobs[:, j])
        if m.sum() < 5:
            continue
        a, b = P[m, j], Yobs[m, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return np.mean(rs) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(y_train.columns)
    gene_set = set(F_train.columns)
    targets = _drug_targets(ctx, cols, gene_set)
    try:
        adj = _load_string(ctx)
    except Exception:
        adj = {}
    nbrs = _drug_neighbours(targets, adj, gene_set)
    Yfill = y_train.fillna(y_train.mean()).fillna(0.0).values
    Yobs = y_train.values

    # inner grouped CV to choose the target / neighbour block weights
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    oof = {c: np.zeros_like(Yfill) for c in GRID}
    for tr, te in splits:
        Ytr_in = y_train.iloc[tr]
        Yf = Ytr_in.fillna(Ytr_in.mean()).fillna(0.0).values
        p = _predict_grid(F_train.iloc[tr], Yf, F_train.iloc[te], targets, nbrs, GRID)
        for c in GRID:
            oof[c][te] = p[c]
    scores = {c: _mean_pearson(oof[c], Yobs) for c in GRID}
    best = max(GRID, key=lambda c: (scores[c], -c[0], -c[1]))

    return _predict_grid(F_train, Yfill, F_test, targets, nbrs, [best])[best]
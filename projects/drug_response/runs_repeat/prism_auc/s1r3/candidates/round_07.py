"""Top-5,000 variable genes, standardized, multi-output ridge (alpha=5000), written in kernel (dual) form
so each drug can get an extra, less-penalized block of its own annotated target genes, plus a shared
block of Hallmark/Reactome gene-set scores (mean z-scored expression). The pathway weight and the
target-block weight are chosen jointly by grouped inner CV (both 0 == original seed model)."""
import os
import re
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]
W_GRID = [0.0, 0.25, 1.0, 4.0]
MIN_SET, MAX_SET = 10, 500

_GS_CACHE = {}


def _gmt_paths(ctx):
    d = ctx.get("genesets_dir")
    gs = ctx.get("genesets")
    cands = []
    if isinstance(gs, dict):
        cands += [v for v in gs.values() if isinstance(v, str)]
    elif isinstance(gs, (list, tuple)):
        cands += [v for v in gs if isinstance(v, str)]
    elif isinstance(gs, str):
        cands.append(gs)
    if d:
        cands = [c if os.path.isabs(c) else os.path.join(str(d), c) for c in cands]
        if os.path.isdir(str(d)):
            cands += [os.path.join(str(d), f) for f in sorted(os.listdir(str(d)))]
    out = []
    for c in cands:
        if (c.endswith(".gmt") or c.endswith(".gmt.gz")) and os.path.isfile(c) and c not in out:
            out.append(c)
    return out


def _load_sets(ctx):
    key = tuple(_gmt_paths(ctx))
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    sets = {}
    for p in key:
        try:
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    genes = sorted({g.strip().upper() for g in parts[2:] if g.strip()})
                    sets[parts[0]] = genes
        except Exception:
            continue
    _GS_CACHE[key] = sets
    return sets


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


def _pathway_scores(F_train, F_test, sets):
    """Mean z-scored expression per gene set (z using training stats), then standardized."""
    if not sets:
        return None, None
    sd_all = F_train.std(ddof=0)
    ok = set(sd_all.index[sd_all > 0.1])
    members = []
    for name, genes in sets.items():
        g = [x for x in genes if x in ok]
        if MIN_SET <= len(g) <= MAX_SET:
            members.append(g)
    if not members:
        return None, None
    ug = sorted({g for m in members for g in m})
    gi = {g: i for i, g in enumerate(ug)}
    M = np.zeros((len(ug), len(members)))
    for j, m in enumerate(members):
        for g in m:
            M[gi[g], j] = 1.0 / len(m)
    mu, sd = F_train[ug].mean(), F_train[ug].std(ddof=0)
    Ztr = ((F_train[ug] - mu) / sd).values
    Zte = ((F_test[ug] - mu) / sd).values
    Ptr, Pte = Ztr @ M, Zte @ M
    pm, ps = Ptr.mean(0), Ptr.std(0)
    ps[ps < 1e-8] = 1.0
    return (Ptr - pm) / ps, (Pte - pm) / ps


def _predict_grid(F_train, Y, F_test, targets, sets, w_grid, s_grid):
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    K0 = Xtr @ Xtr.T
    Kx0 = Xte @ Xtr.T
    n = K0.shape[0]
    ym = Y.mean(0)
    Yc = Y - ym
    Ptr, Pte = _pathway_scores(F_train, F_test, sets)
    if Ptr is not None:
        sc = float(len(genes)) / Ptr.shape[1]
        Kp = sc * (Ptr @ Ptr.T)
        Kxp = sc * (Pte @ Ptr.T)
    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd
    preds = {}
    for w in w_grid:
        if w > 0 and Ptr is None:
            continue
        K = K0 + w * Kp if w > 0 else K0
        Kx = Kx0 + w * Kxp if w > 0 else Kx0
        base = Kx @ np.linalg.solve(K + ALPHA * np.eye(n), Yc) + ym
        for s in s_grid:
            P = base.copy()
            if s > 0 and tg:
                for j, t in enumerate(targets):
                    t = [g for g in t if g in tset]
                    if not t:
                        continue
                    A = Ttr[t].values
                    B = Tte[t].values
                    Kd = K + (s / len(t)) * (A @ A.T)
                    Kxd = Kx + (s / len(t)) * (B @ A.T)
                    P[:, j] = Kxd @ np.linalg.solve(Kd + ALPHA * np.eye(n), Yc[:, j]) + ym[j]
            preds[(w, s)] = P
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
    F_train = F_train.copy()
    F_test = F_test.copy()
    F_train.columns = [str(c).upper() for c in F_train.columns]
    F_test.columns = [str(c).upper() for c in F_test.columns]
    F_train = F_train.loc[:, ~F_train.columns.duplicated()]
    F_test = F_test.loc[:, ~F_test.columns.duplicated()]
    targets = _drug_targets(ctx, cols, set(F_train.columns))
    try:
        sets = _load_sets(ctx)
    except Exception:
        sets = {}
    Yfill = y_train.fillna(y_train.mean()).fillna(0.0).values
    Yobs = y_train.values

    # inner grouped CV to choose pathway-block and target-block weights jointly
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    oof = {}
    for tr, te in splits:
        Ytr_in = y_train.iloc[tr]
        Yf = Ytr_in.fillna(Ytr_in.mean()).fillna(0.0).values
        p = _predict_grid(F_train.iloc[tr], Yf, F_train.iloc[te], targets, sets, W_GRID, S_GRID)
        for k, v in p.items():
            if k not in oof:
                oof[k] = np.zeros_like(Yfill)
            oof[k][te] = v
    scores = {k: _mean_pearson(v, Yobs) for k, v in oof.items()}
    best = max(scores, key=lambda k: (scores[k], -k[0], -k[1]))

    return _predict_grid(F_train, Yfill, F_test, targets, sets, [best[0]], [best[1]])[best]
"""Top-5,000 most variable genes, standardized, plus an optional block of Hallmark/Reactome pathway
activity scores (mean of train-standardized member genes, re-standardized, scaled to the same total
kernel trace as the gene block times gamma). The line x line kernel is the linear kernel of these
features plus eta times a centered Gaussian RBF kernel on the gene block (median-distance bandwidth,
trace-matched to the linear kernel). Multi-output kernel ridge (alpha=5000) on per-drug z-scored AUC
targets, followed by reduced-rank projection of predictions onto the top-k right singular vectors of
the fitted drug-response matrix. On top of the shared low-rank prediction, a per-drug target-gene term
is added (ridge with drug's annotated target genes up-weighted; its difference from the base ridge is
added after projection). Missing training AUCs (z-space) are either zero-filled (r=0) or imputed by
rank-r hard-impute SVD of the training response matrix. (gamma, eta, k, target weight) are chosen
jointly by grouped inner CV on mean per-drug Pearson r; r is chosen in a second stage at best
(gamma, eta)."""
import os
import re
import gzip
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ALPHA = 5000.0
N_GENES = 5000
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 200]
W2_GRID = [0.0, 100.0, 500.0, 2500.0]
G_GRID = [0.0, 0.5, 2.0]
ETA_GRID = [0.0, 0.5, 2.0]
R_GRID = [3, 8, 20]
IMPUTE_ITERS = 50
MAX_TARGETS = 10
MIN_SET, MAX_SET = 10, 500

_GMT_CACHE = {}


def _zscore(Ydf):
    m = np.nanmean(Ydf.values.astype(float), axis=0)
    s = np.nanstd(Ydf.values.astype(float), axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    s = np.where(~np.isfinite(s) | (s < 1e-8), 1.0, s)
    Z = (Ydf.values.astype(float) - m) / s
    Z = np.where(np.isnan(Z), 0.0, Z)
    return Z, m, s


def _impute(Z, obs, r):
    """Hard-impute: replace unobserved entries with a rank-r SVD reconstruction (iterated)."""
    if r <= 0 or obs.all():
        return Z
    X = Z.copy()
    r = min(r, min(X.shape) - 1)
    for _ in range(IMPUTE_ITERS):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :r] * s[:r]) @ Vt[:r]
        Xn = np.where(obs, Z, L)
        if np.abs(Xn - X).max() < 1e-4:
            X = Xn
            break
        X = Xn
    return X


def _prep_targets(Ydf, r):
    Z, m, s = _zscore(Ydf)
    obs = ~np.isnan(Ydf.values.astype(float))
    return _impute(Z, obs, r), m, s


def _lin_kernel(Xtr, Xte):
    xm = Xtr.mean(0)
    A, B = Xtr - xm, Xte - xm
    return A @ A.T, B @ A.T


def _center_kernel(K, Kte):
    rm = K.mean(0)
    tm = rm.mean()
    Kc = K - rm[None, :] - rm[:, None] + tm
    Ktc = Kte - Kte.mean(1)[:, None] - rm[None, :] + tm
    return Kc, Ktc


def _sqdist(A, B):
    a = (A ** 2).sum(1)
    b = (B ** 2).sum(1)
    return np.maximum(a[:, None] + b[None, :] - 2.0 * (A @ B.T), 0.0)


def _kridge(K, Kte, Y, alpha):
    mu = Y.mean(0)
    coef = np.linalg.solve(K + alpha * np.eye(len(K)), Y - mu)
    return mu + K @ coef, mu + Kte @ coef, mu


def _target_deltas(K, Kte, Ttr, Tte, Y, mu, P_te_base, idx_list, alpha, w2s):
    """Per-drug ridge with target-gene kernel up-weighted; returns {w2: delta (n_te x n_drugs)}."""
    tm = Ttr.mean(0)
    Tc, Tt = Ttr - tm, Tte - tm
    n = len(K)
    I = alpha * np.eye(n)
    out = {}
    for w2 in w2s:
        D = np.zeros_like(P_te_base)
        if w2 > 0:
            for j, idx in enumerate(idx_list):
                if not idx:
                    continue
                T, Te = Tc[:, idx], Tt[:, idx]
                c = w2 / len(idx)
                coef = np.linalg.solve(K + c * (T @ T.T) + I, Y[:, j] - mu[j])
                pte = mu[j] + (Kte + c * (Te @ T.T)) @ coef
                D[:, j] = pte - P_te_base[:, j]
        out[w2] = D
    return out


def _project(P_tr, P_te, mu, ks):
    _, _, Vt = np.linalg.svd(P_tr - mu, full_matrices=False)
    out = {}
    for k in ks:
        V = Vt[:min(k, Vt.shape[0])].T
        out[k] = mu + ((P_te - mu) @ V) @ V.T
    return out


def _mean_pearson(P, Yobs):
    rs = []
    for j in range(Yobs.shape[1]):
        m = ~np.isnan(Yobs[:, j])
        if m.sum() < 3:
            continue
        p, y = P[m, j], Yobs[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return np.mean(rs) if rs else 0.0


def _drug_targets(ctx, drug_cols, gene_cols):
    drugs = ctx.get("drugs", None)
    if drugs is None or "target" not in drugs.columns:
        return [[] for _ in drug_cols]
    if "broad_id" in drugs.columns and set(drug_cols).issubset(set(drugs["broad_id"])):
        drugs = drugs.drop_duplicates("broad_id").set_index("broad_id").reindex(drug_cols)
    gset = set(gene_cols)
    out = []
    for t in drugs["target"].values:
        if not isinstance(t, str):
            out.append([])
            continue
        toks = [g.strip().upper() for g in re.split(r"[,|;/]", t)]
        gs = []
        for g in toks:
            if g and g in gset and g not in gs:
                gs.append(g)
        out.append(gs[:MAX_TARGETS])
    return out


def _gmt_paths(ctx):
    d = ctx.get("genesets_dir", None)
    gs = ctx.get("genesets", None)
    cand = []
    if isinstance(gs, dict):
        cand = [v for v in gs.values() if isinstance(v, str)]
    elif isinstance(gs, (list, tuple)):
        cand = [v for v in gs if isinstance(v, str)]
    elif isinstance(gs, str):
        cand = [gs]
    paths = []
    for c in cand:
        p = c if (os.path.isabs(c) or not d) else os.path.join(str(d), c)
        if ".gmt" in p.lower() and os.path.isfile(p) and p not in paths:
            paths.append(p)
    if not paths and d and os.path.isdir(str(d)):
        for f in sorted(os.listdir(str(d))):
            if ".gmt" in f.lower():
                p = os.path.join(str(d), f)
                if os.path.isfile(p):
                    paths.append(p)
    return paths


def _load_gene_sets(ctx):
    paths = tuple(_gmt_paths(ctx))
    if paths in _GMT_CACHE:
        return _GMT_CACHE[paths]
    sets, seen = [], set()
    for p in paths:
        try:
            opener = gzip.open if p.lower().endswith(".gz") else open
            with opener(p, "rt") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    genes = frozenset(g.strip().upper() for g in parts[2:] if g.strip())
                    if genes and genes not in seen:
                        seen.add(genes)
                        sets.append(genes)
        except Exception:
            continue
    _GMT_CACHE[paths] = sets
    return sets


def _pathway_scores(F_train, F_test, valid, ctx):
    sets = _load_gene_sets(ctx)
    if not sets:
        return None, None
    pos = {g.upper(): i for i, g in enumerate(valid)}
    rows, cols, vals = [], [], []
    seen, j = set(), 0
    for s in sets:
        idx = sorted({pos[g] for g in s if g in pos})
        if len(idx) < MIN_SET or len(idx) > MAX_SET:
            continue
        key = tuple(idx)
        if key in seen:
            continue
        seen.add(key)
        rows.extend(idx)
        cols.extend([j] * len(idx))
        vals.extend([1.0 / len(idx)] * len(idx))
        j += 1
    if j < 5:
        return None, None
    M = sparse.csr_matrix((vals, (rows, cols)), shape=(len(valid), j))
    gsc = StandardScaler().fit(F_train[valid].values)
    Str = np.asarray(M.T @ gsc.transform(F_train[valid].values).T).T
    Ste = np.asarray(M.T @ gsc.transform(F_test[valid].values).T).T
    psc = StandardScaler().fit(Str)
    Str, Ste = psc.transform(Str), psc.transform(Ste)
    keep = Str.std(0) > 1e-8
    return Str[:, keep], Ste[:, keep]


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:N_GENES]
    sc = StandardScaler().fit(F_train[genes].values)
    Xg_tr, Xg_te = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yobs = y_train.values.astype(float)
    Z, m_all, s_all = _zscore(y_train)
    has_missing = bool(np.isnan(Yobs).any())

    sd = F_train.std()
    valid = [g for g in F_train.columns if sd[g] > 1e-6]

    # pathway activity block
    try:
        Str, Ste = _pathway_scores(F_train, F_test, valid, ctx)
    except Exception:
        Str, Ste = None, None
    if Str is not None and Str.shape[1] > 0:
        g_grid = G_GRID
        base_scale = Xg_tr.shape[1] / Str.shape[1]
    else:
        g_grid = [0.0]
        base_scale = 0.0

    def build(g):
        if g <= 0:
            return Xg_tr
        return np.hstack([Xg_tr, np.sqrt(g * base_scale) * Str])

    def build_te(g):
        if g <= 0:
            return Xg_te
        return np.hstack([Xg_te, np.sqrt(g * base_scale) * Ste])

    # RBF kernel on the gene block (bandwidth = median squared distance among training lines)
    D_tr = _sqdist(Xg_tr, Xg_tr)
    iu = np.triu_indices(len(D_tr), 1)
    med = np.median(D_tr[iu]) if len(iu[0]) else 1.0
    med = med if med > 1e-8 else 1.0
    R_all = np.exp(-D_tr / med)
    R_te = np.exp(-_sqdist(Xg_te, Xg_tr) / med)
    Rc_all, _ = _center_kernel(R_all, R_all[:1])
    r_trace = max(np.trace(Rc_all), 1e-8)

    # target-gene matrices (standardized on training lines)
    targ = _drug_targets(ctx, list(y_train.columns), valid)
    U = sorted({g for gs in targ for g in gs})
    if U:
        tsc = StandardScaler().fit(F_train[U].values)
        Ttr, Tte = tsc.transform(F_train[U].values), tsc.transform(F_test[U].values)
        pos = {g: i for i, g in enumerate(U)}
        idx_list = [[pos[g] for g in gs] for gs in targ]
        w2_grid = W2_GRID
    else:
        Ttr = np.zeros((len(Xg_tr), 1))
        Tte = np.zeros((len(Xg_te), 1))
        idx_list = [[] for _ in targ]
        w2_grid = [0.0]

    # inner grouped CV to choose pathway weight gamma, RBF weight eta, rank k and target weight w2
    groups = ctx.get("groups_train", None)
    if groups is not None and len(np.unique(groups)) >= 5:
        splits = list(GroupKFold(n_splits=5).split(Xg_tr, groups=np.asarray(groups)))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(Xg_tr))
    rbf_folds = []
    for tr, va in splits:
        rbf_folds.append(_center_kernel(R_all[np.ix_(tr, tr)], R_all[np.ix_(va, tr)]))

    def fold_kernels(g, eta):
        Xall = build(g)
        lin_trace = float(((Xall - Xall.mean(0)) ** 2).sum())
        rscale = lin_trace / r_trace
        out = []
        for f, (tr, va) in enumerate(splits):
            K, Kva = _lin_kernel(Xall[tr], Xall[va])
            if eta > 0:
                Rc, Rcva = rbf_folds[f]
                K = K + eta * rscale * Rc
                Kva = Kva + eta * rscale * Rcva
            out.append((K, Kva))
        return out

    def evaluate(kernels, r, best, best_s, tag):
        oof_proj = {k: np.zeros_like(Z) for k in K_GRID}
        oof_delta = {w: np.zeros_like(Z) for w in w2_grid}
        for f, (tr, va) in enumerate(splits):
            K, Kva = kernels[f]
            Zin, _, _ = _prep_targets(y_train.iloc[tr], r)
            P_tr, P_va, mu = _kridge(K, Kva, Zin, ALPHA)
            proj = _project(P_tr, P_va, mu, K_GRID)
            for k in K_GRID:
                oof_proj[k][va] = proj[k]
            deltas = _target_deltas(K, Kva, Ttr[tr], Ttr[va], Zin, mu, P_va, idx_list, ALPHA, w2_grid)
            for w in w2_grid:
                oof_delta[w][va] = deltas[w]
        for k in K_GRID:
            for w in w2_grid:
                s = _mean_pearson(oof_proj[k] + oof_delta[w], Yobs)
                if s > best_s:
                    best_s, best = s, tag + (k, w)
        return best, best_s

    # stage 1: gamma, eta, k, w2 with zero-filled missing targets
    best, best_s = (g_grid[0], ETA_GRID[0], 0, K_GRID[0], 0.0), -np.inf
    for g in g_grid:
        for eta in ETA_GRID:
            best, best_s = evaluate(fold_kernels(g, eta), 0, best, best_s, (g, eta, 0))

    # stage 2: low-rank imputation of missing training AUCs at best (gamma, eta)
    if has_missing:
        g0, e0 = best[0], best[1]
        kernels = fold_kernels(g0, e0)
        for r in R_GRID:
            best, best_s = evaluate(kernels, r, best, best_s, (g0, e0, r))
    best_g, best_eta, best_r, best_k, best_w = best

    Xtr, Xte = build(best_g), build_te(best_g)
    K, Kte = _lin_kernel(Xtr, Xte)
    if best_eta > 0:
        lin_trace = float(((Xtr - Xtr.mean(0)) ** 2).sum())
        rscale = lin_trace / r_trace
        Rc, Rcte = _center_kernel(R_all, R_te)
        K = K + best_eta * rscale * Rc
        Kte = Kte + best_eta * rscale * Rcte
    Zfit, _, _ = _prep_targets(y_train, best_r)
    P_tr, P_te, mu = _kridge(K, Kte, Zfit, ALPHA)
    Pz = _project(P_tr, P_te, mu, [best_k])[best_k]
    if best_w > 0:
        Pz = Pz + _target_deltas(K, Kte, Ttr, Tte, Zfit, mu, P_te, idx_list, ALPHA, [best_w])[best_w]
    return Pz * s_all + m_all
"""Top-5,000 most variable genes, standardized, plus an optional block of Hallmark/Reactome pathway
activity scores (mean of train-standardized member genes, re-standardized, scaled to the same total
kernel trace as the gene block times gamma). The line x line kernel is the linear kernel of these
features plus eta times a centered Gaussian RBF kernel on the gene block (median-distance bandwidth,
trace-matched to the linear kernel). Multi-output kernel ridge (alpha=5000) on per-drug transformed AUC
targets (plain z-score, z-score winsorized at +-2.5, or rank-to-Gaussian of observed values), followed
by soft reduced-rank projection: predictions are projected onto the top-k right singular vectors of the
fitted drug-response matrix, and a fraction lam of the out-of-subspace residual is kept (lam=0 is hard
truncation). On top of the shared low-rank prediction, a per-drug target-gene term is added (ridge with
drug's annotated target genes up-weighted; its difference from the base ridge is added after
projection). Finally, each drug's z-space prediction is mixed (weight rho) with the mean prediction of
its annotated neighbours (Jaccard overlap of MOA tokens and target genes). Missing training AUCs
(z-space) are either zero-filled (r=0) or imputed by rank-r hard-impute SVD of the training response
matrix. (gamma, eta, k, lam, target weight, rho) are chosen jointly by grouped inner CV on mean per-drug
Pearson r; r is chosen in a second stage at best (gamma, eta); the target transform in a third stage.
The top-10 inner-CV configurations are retained; the final prediction averages the top-M of them (z-space),
with M chosen by the inner-CV score of the averaged out-of-fold predictions (M=1 = single best)."""
import os
import re
import gzip
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, norm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ALPHA = 5000.0
N_GENES = 5000
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 200]
LAM_GRID = [0.0, 0.2, 0.5]
W2_GRID = [0.0, 100.0, 500.0, 2500.0]
G_GRID = [0.0, 0.5, 2.0]
ETA_GRID = [0.0, 0.5, 2.0]
R_GRID = [3, 8, 20]
RHO_GRID = [0.0, 0.25, 0.5]
T_GRID = ["clip", "rg"]
M_GRID = [1, 2, 3, 5, 8]
TOPN = 10
CLIP = 2.5
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


def _transform(Ydf, t):
    """Per-drug target transform in z-space; missing entries are 0."""
    Z, m, s = _zscore(Ydf)
    Y = Ydf.values.astype(float)
    obs = ~np.isnan(Y)
    if t == "clip":
        Z = np.clip(Z, -CLIP, CLIP)
    elif t == "rg":
        for j in range(Z.shape[1]):
            o = obs[:, j]
            n = int(o.sum())
            if n >= 2:
                rk = rankdata(Y[o, j])
                Z[o, j] = norm.ppf((rk - 0.5) / n)
    return Z, obs, m, s


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


def _prep_targets(Ydf, r, t="z"):
    Z, obs, m, s = _transform(Ydf, t)
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


def _project(P_tr, P_te, mu, ks, lams):
    """Soft reduced-rank projection: top-k subspace plus lam times the out-of-subspace residual."""
    _, _, Vt = np.linalg.svd(P_tr - mu, full_matrices=False)
    D = P_te - mu
    out = {}
    for k in ks:
        V = Vt[:min(k, Vt.shape[0])].T
        Pk = (D @ V) @ V.T
        for lam in lams:
            out[(k, lam)] = mu + Pk + lam * (D - Pk)
    return out


def _mean_pearson(P, Yobs):
    """Vectorized mean over drugs of Pearson r on observed entries."""
    M = ~np.isnan(Yobs)
    Mf = M.astype(float)
    Y0 = np.where(M, Yobs, 0.0)
    n = Mf.sum(0)
    nn = np.maximum(n, 1.0)
    Pm = P * Mf
    sp, sy = Pm.sum(0), Y0.sum(0)
    spp, syy, spy = (Pm * P).sum(0), (Y0 * Y0).sum(0), (Pm * Y0).sum(0)
    cov = spy - sp * sy / nn
    vp = spp - sp * sp / nn
    vy = syy - sy * sy / nn
    ok = (n >= 3) & (vp > nn * 1e-24) & (vy > nn * 1e-24)
    if not ok.any():
        return 0.0
    r = cov[ok] / np.sqrt(vp[ok] * vy[ok])
    return float(np.mean(r))


def _drug_table(ctx, drug_cols):
    drugs = ctx.get("drugs", None)
    if drugs is None:
        return None
    if "broad_id" in drugs.columns and set(drug_cols).issubset(set(drugs["broad_id"])):
        drugs = drugs.drop_duplicates("broad_id").set_index("broad_id").reindex(drug_cols)
    if len(drugs) != len(drug_cols):
        return None
    return drugs


def _drug_targets(ctx, drug_cols, gene_cols):
    drugs = _drug_table(ctx, drug_cols)
    if drugs is None or "target" not in drugs.columns:
        return [[] for _ in drug_cols]
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


def _tokens(v, pat):
    if not isinstance(v, str):
        return set()
    return {t.strip().upper() for t in re.split(pat, v) if t.strip()}


def _drug_neighbour_matrix(ctx, drug_cols):
    """Row-stochastic drug x drug mixing matrix from MOA / target Jaccard overlap (self excluded);
    drugs with no annotated neighbour map to themselves."""
    nd = len(drug_cols)
    W = np.eye(nd)
    drugs = _drug_table(ctx, drug_cols)
    if drugs is None:
        return W, False
    moas = [_tokens(v, r"[,|;]") for v in (drugs["moa"].values if "moa" in drugs.columns
                                             else [None] * nd)]
    tgts = [_tokens(v, r"[,|;/]") for v in (drugs["target"].values if "target" in drugs.columns
                                              else [None] * nd)]

    def jac(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    S = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(i + 1, nd):
            v = 0.5 * jac(moas[i], moas[j]) + 0.5 * jac(tgts[i], tgts[j])
            S[i, j] = S[j, i] = v
    rs = S.sum(1)
    any_nb = False
    for i in range(nd):
        if rs[i] > 1e-12:
            W[i] = S[i] / rs[i]
            any_nb = True
    return W, any_nb


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

    # MOA / target neighbour mixing matrix
    try:
        Wnb, any_nb = _drug_neighbour_matrix(ctx, list(y_train.columns))
    except Exception:
        Wnb, any_nb = np.eye(Z.shape[1]), False
    rho_grid = RHO_GRID if any_nb else [0.0]
    WnbT = Wnb.T

    # inner grouped CV to choose gamma, eta, k, lam, w2, rho, r and target transform
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

    proj_keys = [(k, lam) for k in K_GRID for lam in LAM_GRID]
    top = []  # (score, config, out-of-fold z-space predictions), best first

    def push(s, cfg, P):
        if len(top) < TOPN or s > top[-1][0]:
            top.append((s, cfg, P.copy()))
            top.sort(key=lambda x: -x[0])
            del top[TOPN:]

    def evaluate(kernels, r, t, best, best_s, tag):
        oof_proj = {key: np.zeros_like(Z) for key in proj_keys}
        oof_delta = {w: np.zeros_like(Z) for w in w2_grid}
        for f, (tr, va) in enumerate(splits):
            K, Kva = kernels[f]
            Zin, _, _ = _prep_targets(y_train.iloc[tr], r, t)
            P_tr, P_va, mu = _kridge(K, Kva, Zin, ALPHA)
            proj = _project(P_tr, P_va, mu, K_GRID, LAM_GRID)
            for key in proj_keys:
                oof_proj[key][va] = proj[key]
            deltas = _target_deltas(K, Kva, Ttr[tr], Ttr[va], Zin, mu, P_va, idx_list, ALPHA, w2_grid)
            for w in w2_grid:
                oof_delta[w][va] = deltas[w]
        for key in proj_keys:
            for w in w2_grid:
                base = oof_proj[key] + oof_delta[w]
                sm = base @ WnbT if any_nb else base
                for rho in rho_grid:
                    P = base if rho == 0 else (1.0 - rho) * base + rho * sm
                    s = _mean_pearson(P, Yobs)
                    cfg = tag + key + (w, rho)
                    push(s, cfg, P)
                    if s > best_s:
                        best_s, best = s, cfg
        return best, best_s

    # stage 1: gamma, eta, k, lam, w2, rho with zero-filled missing targets
    best, best_s = (g_grid[0], ETA_GRID[0], 0, "z", K_GRID[0], 0.0, 0.0, 0.0), -np.inf
    for g in g_grid:
        for eta in ETA_GRID:
            best, best_s = evaluate(fold_kernels(g, eta), 0, "z", best, best_s, (g, eta, 0, "z"))

    # stage 2: low-rank imputation of missing training AUCs at best (gamma, eta)
    g0, e0 = best[0], best[1]
    kernels = fold_kernels(g0, e0)
    if has_missing:
        for r in R_GRID:
            best, best_s = evaluate(kernels, r, "z", best, best_s, (g0, e0, r, "z"))

    # stage 3: robust target transform at best (gamma, eta, r)
    r0 = best[2]
    for t in T_GRID:
        best, best_s = evaluate(kernels, r0, t, best, best_s, (g0, e0, r0, t))
    del kernels

    # stage 4: number of top configurations to average (M=1 is the single best)
    configs = [best]
    if top:
        best_m_s = -np.inf
        for M in M_GRID:
            if M > len(top):
                break
            s = _mean_pearson(np.mean([e[2] for e in top[:M]], axis=0), Yobs)
            if s > best_m_s + 1e-12:
                best_m_s, configs = s, [e[1] for e in top[:M]]
    top.clear()

    # final refit of each selected configuration on the full training fold
    kcache, zcache = {}, {}

    def full_kernel(g, eta):
        if (g, eta) not in kcache:
            Xtr, Xte = build(g), build_te(g)
            K, Kte = _lin_kernel(Xtr, Xte)
            if eta > 0:
                lin_trace = float(((Xtr - Xtr.mean(0)) ** 2).sum())
                rscale = lin_trace / r_trace
                Rc, Rcte = _center_kernel(R_all, R_te)
                K = K + eta * rscale * Rc
                Kte = Kte + eta * rscale * Rcte
            kcache[(g, eta)] = (K, Kte)
        return kcache[(g, eta)]

    preds = []
    for cfg in configs:
        g, eta, r, t, k, lam, w, rho = cfg
        K, Kte = full_kernel(g, eta)
        if (r, t) not in zcache:
            zcache[(r, t)] = _prep_targets(y_train, r, t)[0]
        Zfit = zcache[(r, t)]
        P_tr, P_te, mu = _kridge(K, Kte, Zfit, ALPHA)
        Pz = _project(P_tr, P_te, mu, [k], [lam])[(k, lam)]
        if w > 0:
            Pz = Pz + _target_deltas(K, Kte, Ttr, Tte, Zfit, mu, P_te, idx_list, ALPHA, [w])[w]
        if rho > 0:
            Pz = (1.0 - rho) * Pz + rho * (Pz @ WnbT)
        preds.append(Pz)
    Pz = np.mean(preds, axis=0)
    return Pz * s_all + m_all
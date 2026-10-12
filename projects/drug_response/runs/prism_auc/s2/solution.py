"""Multi-scale, multi-view ensemble of shared multi-output ridges (variance-filtered gene pools, multi-task
supervised pools of genes ranked by mean squared correlation with all drugs' AUCs, and drug-cluster-specific
supervised pools whose ridges only contribute to the drugs in that response cluster, x penalties x
{z-scored, variance-weighted (unscaled centred), rank-Gaussianised} gene views x {raw, per-drug
rank-Gaussianised} targets, averaged as z-scored predictions) blended with an ensemble of per-drug ridges on
drug-specific correlation-selected genes (averaged over several panel sizes / penalties), then smoothed across
drugs with similar response profiles (kNN in drug-response-correlation space); blend weight / smoothing
strength chosen by grouped inner CV."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import norm, rankdata
from sklearn.model_selection import GroupKFold, KFold

POOL = 8000
KS = (50, 100, 250, 500, 1000)
LAMS = (0.5, 2.0, 8.0)       # alpha = lam * k
BASE_POOLS = (2000, 5000, 10000)
SUP_CAND = 15000             # candidate genes (by variance) for the multi-task supervised pools
SUP_POOLS = (1000, 3000)     # top genes by mean squared correlation across all drugs
NCL = 10                     # drug response clusters for cluster-specific supervised pools
CL_POOLS = (500, 2000)       # top genes by mean squared correlation within a drug cluster
BASE_A = (0.5, 1.0, 2.0)     # shared-ridge alpha = a * mean(diag(K)) (= a * n_genes for z-scored genes)
BASE_VIEWS = ("z", "c", "rg")  # z-scored, centred-only (variance-weighted), rank-Gaussianised genes
WS = (0.0, 0.33, 0.5, 0.67, 1.0)  # weight on the per-drug filtered ensemble
GS = (0.0, 0.25, 0.5)        # weight on the drug-neighbour-smoothed prediction
NNB = 10                     # neighbours per drug
POW = 2.0                    # similarity sharpening
RG = "rg|"


def featurize(items, ctx):
    """Raw log2(TPM+1) expression plus a label-free per-gene rank-Gaussianised copy (outlier-robust view)."""
    X = items.astype(float)
    X.columns = [str(c) for c in X.columns]
    n = len(X)
    R = X.rank(axis=0, method="average").values
    G = pd.DataFrame(norm.ppf((R - 0.5) / n), index=X.index, columns=[RG + c for c in X.columns])
    return pd.concat([X, G], axis=1)


def _split(F):
    cols = pd.Index([str(c) for c in F.columns])
    m = np.asarray(cols.str.startswith(RG))
    if not m.any():
        return F, None
    return F.loc[:, ~m], F.loc[:, m]


def _gauss_targets(Y):
    """Per-drug rank-Gaussianised observed AUCs (NaN -> 0 = the drug's mean), centred."""
    G = np.zeros_like(Y)
    for d in range(Y.shape[1]):
        obs = ~np.isnan(Y[:, d])
        m = obs.sum()
        if m < 2:
            continue
        r = rankdata(Y[obs, d])
        G[obs, d] = norm.ppf((r - 0.5) / m)
    return G - G.mean(0)


def _drug_clusters(Yz):
    """Ward clustering of drugs on their (unit-norm) training response profiles."""
    D = Yz.shape[1]
    if D <= NCL:
        return np.arange(D)
    P = Yz.T / (np.linalg.norm(Yz.T, axis=1, keepdims=True) + 1e-12)
    Z = linkage(P, method="ward")
    return fcluster(Z, t=NCL, criterion="maxclust")


def _base(Ftr, Rtr, Y, Fte, Rte):
    """Average of z-scored multi-output ridge predictions over gene pools x views x penalties x target views."""
    order = Ftr.var().sort_values(ascending=False).index
    Yf = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    Yf = np.nan_to_num(Yf)
    Yc = Yf - Yf.mean(0)
    Yg = _gauss_targets(Y)
    D = Y.shape[1]
    allc = np.arange(D)
    pools = [(order[:min(p, len(order))], allc) for p in BASE_POOLS]
    # multi-task supervised pools: genes most correlated with drug response on average across drugs
    cand = order[:min(SUP_CAND, len(order))]
    Xs = Ftr[cand].values
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    Yz = Yc / (Yc.std(0) + 1e-12)
    G2 = ((Xs.T @ Yz) / Xs.shape[0]) ** 2
    del Xs
    so = np.argsort(-G2.mean(1))
    for p in SUP_POOLS:
        pools.append((cand[so[:min(p, len(cand))]], allc))
    # cluster-specific supervised pools: genes tied to a group of response-similar drugs
    labels = _drug_clusters(Yz)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        oc = np.argsort(-G2[:, idx].mean(1))
        for p in CL_POOLS:
            pools.append((cand[oc[:min(p, len(cand))]], idx))
    del G2
    acc = np.zeros((len(Fte), D))
    for genes, cols in pools:
        for view in BASE_VIEWS:
            if view == "rg":
                if Rtr is None:
                    continue
                rg = [RG + str(g) for g in genes]
                Xtr0, Xte0 = Rtr[rg].values, Rte[rg].values
            else:
                Xtr0, Xte0 = Ftr[genes].values, Fte[genes].values
            mu = Xtr0.mean(0)
            if view == "c":
                Xtr, Xte = Xtr0 - mu, Xte0 - mu
            else:
                sd = Xtr0.std(0) + 1e-8
                Xtr, Xte = (Xtr0 - mu) / sd, (Xte0 - mu) / sd
            K = Xtr @ Xtr.T
            kscale = np.trace(K) / K.shape[0]
            ev, U = np.linalg.eigh(K)
            ev = np.clip(ev, 0, None)
            KU = (Xte @ Xtr.T) @ U
            for T in (Yc, Yg):
                UY = U.T @ T[:, cols]
                for a in BASE_A:
                    P = KU @ (UY / (ev + a * kscale)[:, None])
                    acc[:, cols] += _z(P)
    return acc


def _filtered(Ftr, Y, Fte):
    """Average of z-scored per-drug ridge predictions over all (k, lam) configurations."""
    genes = Ftr.var().sort_values(ascending=False).index[:POOL]
    Xtr, Xte = Ftr[genes].values, Fte[genes].values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    D = Y.shape[1]
    out = np.zeros((Xte.shape[0], D))
    for d in range(D):
        obs = ~np.isnan(Y[:, d])
        if obs.sum() < 10:
            continue
        yo = Y[obs, d]
        ym = yo.mean()
        Xo = Xtr[obs]
        xm = Xo.mean(0)
        Xc = Xo - xm
        yc = yo - ym
        r = (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-8)
        order = np.argsort(-np.abs(r))
        acc = np.zeros(Xte.shape[0])
        for k in KS:
            idx = order[:k]
            A = Xc[:, idx]
            At = Xte[:, idx] - xm[idx]
            ev, U = np.linalg.eigh(A @ A.T)
            ev = np.clip(ev, 0, None)
            Uy = U.T @ yc
            Kte = At @ A.T
            for l in LAMS:
                p = Kte @ (U @ (Uy / (ev + l * k)))
                s = p.std()
                if s > 1e-12:
                    acc += (p - p.mean()) / s
        out[:, d] = acc
    return out


def _neighbours(Y):
    """Row-normalised drug x drug weight matrix from response correlations (training rows only)."""
    D = Y.shape[1]
    Yf = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    Yf = np.nan_to_num(Yf)
    Yz = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12)
    C = (Yz.T @ Yz) / Yz.shape[0]
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, 0, None) ** POW
    W = np.zeros((D, D))
    for d in range(D):
        nb = np.argsort(-C[d])[:NNB]
        w = C[d, nb]
        s = w.sum()
        if s > 1e-12:
            W[d, nb] = w / s
        else:
            W[d, d] = 1.0
    return W


def _smooth(P, W, g):
    if g == 0.0:
        return P
    return (1 - g) * P + g * (P @ W.T)


def _z(P):
    return (P - P.mean(0)) / (P.std(0) + 1e-12)


def _mean_pearson(P, Y):
    rs = []
    for d in range(Y.shape[1]):
        m = ~np.isnan(Y[:, d])
        if m.sum() < 3:
            continue
        a, b = P[m, d], Y[m, d]
        if a.std() < 1e-12 or b.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    Y = y_train.values.astype(float)
    n = len(F_train)
    Xtr_all, Rtr_all = _split(F_train)
    Xte_all, Rte_all = _split(F_test)
    groups = ctx.get("groups_train", None)
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = list(GroupKFold(n_splits=3).split(np.zeros(n), groups=np.asarray(groups)))
    else:
        splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(n)))

    # inner CV: collect OOF z-scored predictions
    oof_base = np.zeros_like(Y)
    oof_filt = np.zeros_like(Y)
    fold_W = []
    for tr, te in splits:
        Ftr, Fte = Xtr_all.iloc[tr], Xtr_all.iloc[te]
        Rtr = Rtr_all.iloc[tr] if Rtr_all is not None else None
        Rte = Rtr_all.iloc[te] if Rtr_all is not None else None
        oof_base[te] = _z(_base(Ftr, Rtr, Y[tr], Fte, Rte))
        oof_filt[te] = _z(_filtered(Ftr, Y[tr], Fte))
        fold_W.append(_neighbours(Y[tr]))

    best, best_s = (0.0, 0.0), -np.inf
    for w in WS:
        B = (1 - w) * oof_base + w * oof_filt
        for g in GS:
            if g == 0.0:
                Bs = B
            else:
                Bs = np.zeros_like(B)
                for (tr, te), Wf in zip(splits, fold_W):
                    Bs[te] = _smooth(B[te], Wf, g)
            s = _mean_pearson(Bs, Y)
            if s > best_s:
                best_s, best = s, (w, g)
    w, g = best

    Z = _z(_base(Xtr_all, Rtr_all, Y, Xte_all, Rte_all))
    if w > 0.0:
        Pf = _filtered(Xtr_all, Y, Xte_all)
        Z = (1 - w) * Z + w * _z(Pf)
    if g > 0.0:
        Z = _z(_smooth(Z, _neighbours(Y), g))
    else:
        Z = _z(Z)
    # rescale to the training AUC scale (Pearson is invariant; keeps outputs in AUC units)
    ym = np.nan_to_num(np.nanmean(Y, 0))
    ys = np.nan_to_num(np.nanstd(Y, 0))
    return Z * (0.3 * ys + 1e-6) + ym
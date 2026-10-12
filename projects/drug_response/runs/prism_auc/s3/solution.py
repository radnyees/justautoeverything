"""Top-5,000 most variable genes, standardized, multi-output ridge (alpha=5000, dual/kernel form)
on per-drug z-scored AUC targets, followed by reduced-rank projection of predictions onto the
top-k right singular vectors of the fitted (standardized) drug-response matrix (k chosen by
grouped inner CV on mean per-drug Pearson r). Standardizing targets lets every drug contribute
equally to the shared response axes instead of high-variance drugs dominating them."""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ALPHA = 5000.0
K_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 200]


def _zscore(Ydf):
    m = np.nanmean(Ydf.values.astype(float), axis=0)
    s = np.nanstd(Ydf.values.astype(float), axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    s = np.where(~np.isfinite(s) | (s < 1e-8), 1.0, s)
    Z = (Ydf.values.astype(float) - m) / s
    Z = np.where(np.isnan(Z), 0.0, Z)
    return Z, m, s


def _ridge(Xtr, Y, Xte, alpha):
    mu = Y.mean(0)
    xm = Xtr.mean(0)
    A, B = Xtr - xm, Xte - xm
    K = A @ A.T
    coef = np.linalg.solve(K + alpha * np.eye(len(K)), Y - mu)
    return mu + K @ coef, mu + (B @ A.T) @ coef, mu


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


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yobs = y_train.values.astype(float)
    Z, m_all, s_all = _zscore(y_train)

    # inner grouped CV to choose the rank k
    groups = ctx.get("groups_train", None)
    if groups is not None and len(np.unique(groups)) >= 5:
        splits = list(GroupKFold(n_splits=5).split(Xtr, groups=np.asarray(groups)))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(Xtr))
    oof = {k: np.zeros_like(Z) for k in K_GRID}
    for tr, va in splits:
        Zin, _, _ = _zscore(y_train.iloc[tr])
        P_tr, P_va, mu = _ridge(Xtr[tr], Zin, Xtr[va], ALPHA)
        proj = _project(P_tr, P_va, mu, K_GRID)
        for k in K_GRID:
            oof[k][va] = proj[k]
    scores = {k: _mean_pearson(oof[k], Yobs) for k in K_GRID}
    best_k = max(K_GRID, key=lambda k: scores[k])

    P_tr, P_te, mu = _ridge(Xtr, Z, Xte, ALPHA)
    Pz = _project(P_tr, P_te, mu, [best_k])[best_k]
    return Pz * s_all + m_all
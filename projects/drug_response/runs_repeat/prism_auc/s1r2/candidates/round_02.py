"""Top-5,000 most variable genes, standardized, kernel-form ridge (alpha=5000) predicting a
low-rank (iteratively SVD-imputed) version of the drug x line AUC matrix; rank chosen by grouped
inner CV on the training fold (reduced-rank multi-task regression)."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHA = 5000.0
RANKS = [3, 5, 10, 20, 40, 80, None]  # None = no low-rank (mean fill, = seed model)


def _lowrank(Y, mask, k, iters=20):
    mu = np.nanmean(Y, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Z = np.where(mask, Y, mu)
    if k is None:
        return Z
    for _ in range(iters):
        m = Z.mean(axis=0)
        U, s, Vt = np.linalg.svd(Z - m, full_matrices=False)
        R = (U[:, :k] * s[:k]) @ Vt[:k] + m
        Z = np.where(mask, Y, R)
    m = Z.mean(axis=0)
    U, s, Vt = np.linalg.svd(Z - m, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k] + m


def _prep(Ftr, Fte):
    genes = Ftr.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(Ftr[genes].values)
    return sc.transform(Ftr[genes].values), sc.transform(Fte[genes].values)


def _ridge_dual(Xtr, T, Xte):
    K = Xtr @ Xtr.T
    tm = T.mean(axis=0)
    A = np.linalg.solve(K + ALPHA * np.eye(len(K)), T - tm)
    return (Xte @ Xtr.T) @ A + tm


def _score(P, Y, mask):
    rs = []
    for j in range(Y.shape[1]):
        m = mask[:, j]
        if m.sum() < 3:
            continue
        p, y = P[m, j], Y[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return np.mean(rs) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    Y = y_train.values.astype(float)
    mask = np.isfinite(Y)
    groups = np.asarray(ctx.get("groups_train", np.arange(len(Y))))

    # inner grouped CV to choose the rank of the target matrix
    scores = {i: [] for i in range(len(RANKS))}
    gkf = GroupKFold(n_splits=3)
    for itr, iva in gkf.split(F_train.values, groups=groups):
        Xa, Xb = _prep(F_train.iloc[itr], F_train.iloc[iva])
        Ya, ma = Y[itr], mask[itr]
        for i, k in enumerate(RANKS):
            T = _lowrank(Ya, ma, k)
            P = _ridge_dual(Xa, T, Xb)
            scores[i].append(_score(P, Y[iva], mask[iva]))
    best = max(scores, key=lambda i: np.mean(scores[i]))
    k = RANKS[best]

    Xtr, Xte = _prep(F_train, F_test)
    T = _lowrank(Y, mask, k)
    return _ridge_dual(Xtr, T, Xte)
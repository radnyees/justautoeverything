"""Top-5,000 most variable genes, standardized. Blend of a multi-output linear ridge and a multi-output
RBF kernel ridge; the kernel penalty and blend weight are chosen by grouped inner CV on the training fold."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

KRR_ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _mean_pearson(Yobs, P):
    rs = []
    for j in range(Yobs.shape[1]):
        m = ~np.isnan(Yobs[:, j])
        if m.sum() < 3:
            continue
        a, b = Yobs[m, j], P[m, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return np.mean(rs) if rs else 0.0


def _ridge(Xtr, Y, Xte):
    return Ridge(alpha=5000.0).fit(Xtr, Y).predict(Xte)


def _krr_multi(Xtr, Y, Xte, alphas):
    D = cdist(Xtr, Xtr, "sqeuclidean")
    gamma = 1.0 / np.median(D[np.triu_indices_from(D, k=1)])
    K = np.exp(-gamma * D)
    Kte = np.exp(-gamma * cdist(Xte, Xtr, "sqeuclidean"))
    mu = Y.mean(0)
    s, U = np.linalg.eigh(K)
    s = np.clip(s, 0, None)
    UtY = U.T @ (Y - mu)
    KteU = Kte @ U
    return {a: KteU @ (UtY / (s + a)[:, None]) + mu for a in alphas}


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    Y = y_train.fillna(y_train.mean()).values

    # grouped inner CV to choose kernel alpha and blend weight
    groups = np.asarray(ctx.get("groups_train", np.arange(len(Xtr))))
    n_splits = min(3, len(np.unique(groups)))
    oof_r = np.zeros_like(Y)
    oof_k = {a: np.zeros_like(Y) for a in KRR_ALPHAS}
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, groups=groups):
        Yi = y_train.iloc[tr]
        Yi = Yi.fillna(Yi.mean()).fillna(0.0).values
        oof_r[va] = _ridge(Xtr[tr], Yi, Xtr[va])
        pk = _krr_multi(Xtr[tr], Yi, Xtr[va], KRR_ALPHAS)
        for a in KRR_ALPHAS:
            oof_k[a][va] = pk[a]

    best = (_mean_pearson(Yraw, oof_r), 0.0, KRR_ALPHAS[0])
    for a in KRR_ALPHAS:
        for w in WEIGHTS[1:]:
            sc_ = _mean_pearson(Yraw, (1 - w) * oof_r + w * oof_k[a])
            if sc_ > best[0]:
                best = (sc_, w, a)
    _, w, a = best

    pred = _ridge(Xtr, Y, Xte)
    if w > 0:
        pred = (1 - w) * pred + w * _krr_multi(Xtr, Y, Xte, [a])[a]
    return pred
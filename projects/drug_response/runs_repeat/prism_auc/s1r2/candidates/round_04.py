"""Top-5,000 most variable genes, standardized. Blend of (a) multi-output linear ridge (alpha=5000, the
previous champion) and (b) multi-output RBF kernel ridge capturing nonlinear cell-line similarity.
Blend weight and kernel penalty chosen by grouped inner CV on mean per-drug Pearson r."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

LIN_ALPHA = 5000.0
RBF_ALPHAS = [0.03, 0.1, 0.3, 1.0]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _lin_fit_predict(Xa, Ya, Xb):
    mu = Xa.mean(0)
    A, B = Xa - mu, Xb - mu
    ym = Ya.mean(0)
    K = A @ A.T
    dual = np.linalg.solve(K + LIN_ALPHA * np.eye(len(A)), Ya - ym)
    return (B @ A.T) @ dual + ym


def _sqdist(A, B):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _rbf_fit_predict(Xa, Ya, Xb, alphas):
    Daa = _sqdist(Xa, Xa)
    iu = np.triu_indices(len(Xa), 1)
    gamma = 1.0 / np.median(Daa[iu])
    Kaa = np.exp(-gamma * Daa)
    Kba = np.exp(-gamma * _sqdist(Xb, Xa))
    ym = Ya.mean(0)
    out = []
    for a in alphas:
        dual = np.linalg.solve(Kaa + a * np.eye(len(Xa)), Ya - ym)
        out.append(Kba @ dual + ym)
    return out


def _mean_pearson(P, Y):
    rs = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 3:
            continue
        p, y = P[m, j], Y[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr = sc.transform(F_train[genes].values)
    Xte = sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    Y = y_train.fillna(y_train.mean()).values

    # inner grouped CV to choose blend weight and RBF penalty
    groups = np.asarray(ctx.get("groups_train", np.arange(len(Xtr))))
    n_splits = min(3, len(np.unique(groups)))
    oof_lin = np.zeros_like(Y)
    oof_rbf = [np.zeros_like(Y) for _ in RBF_ALPHAS]
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, Y, groups):
        Ytr_in = np.asarray(y_train.iloc[tr].fillna(y_train.iloc[tr].mean()).values, dtype=float)
        Ytr_in = np.where(np.isnan(Ytr_in), np.nanmean(Yraw, axis=0), Ytr_in)
        oof_lin[va] = _lin_fit_predict(Xtr[tr], Ytr_in, Xtr[va])
        preds = _rbf_fit_predict(Xtr[tr], Ytr_in, Xtr[va], RBF_ALPHAS)
        for k, p in enumerate(preds):
            oof_rbf[k][va] = p

    best = (-np.inf, 0.0, 0)
    for k in range(len(RBF_ALPHAS)):
        for w in WEIGHTS:
            s = _mean_pearson((1 - w) * oof_lin + w * oof_rbf[k], Yraw)
            if s > best[0] + 1e-9:
                best = (s, w, k)
    _, w, k = best

    pred = _lin_fit_predict(Xtr, Y, Xte)
    if w > 0:
        prbf = _rbf_fit_predict(Xtr, Y, Xte, [RBF_ALPHAS[k]])[0]
        pred = (1 - w) * pred + w * prbf
    return pred
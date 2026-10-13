"""Tangent-space embedding of Ledoit-Wolf covariances on CC200, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)) @ V.T


def _riemann_mean(covs, n_iter=20, tol=1e-7):
    # init with log-euclidean mean
    M = _eig_fn(np.mean([_eig_fn(c, np.log) for c in covs], axis=0), np.exp)
    for _ in range(n_iter):
        Mh = _eig_fn(M, np.sqrt)
        Mih = _eig_fn(M, lambda w: 1.0 / np.sqrt(w))
        T = np.mean([_eig_fn(Mih @ c @ Mih, np.log) for c in covs], axis=0)
        M = Mh @ _eig_fn(T, np.exp) @ Mh
        if np.linalg.norm(T) < tol:
            break
    return M


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0
        x = (x - x.mean(axis=0)) / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)
    M = _riemann_mean(covs)
    Mih = _eig_fn(M, lambda w: 1.0 / np.sqrt(w))
    n = M.shape[0]
    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        L = _eig_fn(Mih @ c @ Mih, np.log)
        feats[i] = (coef * L[iu]).astype(np.float32)
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
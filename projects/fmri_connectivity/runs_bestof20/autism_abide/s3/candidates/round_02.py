"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _standardize(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return x / sd


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        z = _standardize(x)
        c = LedoitWolf(assume_centered=True).fit(z).covariance_
        # guarantee positive definiteness (all-zero ROIs give zero rows)
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c)
    return np.asarray(covs, dtype=np.float32)


def _eig_fun(C, fun):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * fun(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _tangent(covs, W, iu, coef):
    out = np.empty((covs.shape[0], len(iu[0])), dtype=np.float32)
    for i in range(covs.shape[0]):
        M = W @ covs[i] @ W
        M = (M + M.T) / 2.0
        L = _eig_fun(M, np.log)
        out[i] = (L[iu] * coef).astype(np.float32)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    # log-Euclidean mean of training covariances as tangent reference
    logmean = np.mean([_eig_fun(c, np.log) for c in Ctr], axis=0)
    logmean = (logmean + logmean.T) / 2.0
    ref = _eig_fun(logmean, np.exp)
    W = _eig_fun(ref, lambda w: 1.0 / np.sqrt(w))
    n = Ctr.shape[1]
    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    Xtr = _tangent(Ctr, W, iu, coef)
    Xte = _tangent(Cte, W, iu, coef)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
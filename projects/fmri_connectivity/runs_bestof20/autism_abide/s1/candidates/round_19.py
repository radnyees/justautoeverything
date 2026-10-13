"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps the matrix PD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        covs.append(c.astype(np.float32))
    return np.stack(covs)


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)) @ V.T


def _tangent(covs, ref_isqrt, iu, coef):
    out = np.empty((len(covs), len(iu[0])), dtype=np.float64)
    for i, C in enumerate(covs):
        W = ref_isqrt @ C.astype(np.float64) @ ref_isqrt
        W = (W + W.T) / 2
        L = _eig_fn(W, np.log)
        out[i] = L[iu] * coef
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    p = F_train.shape[1]
    # Log-Euclidean mean of training covariances as tangent reference point
    logmean = np.zeros((p, p))
    for C in F_train:
        logmean += _eig_fn(C.astype(np.float64), np.log)
    logmean /= len(F_train)
    ref = _eig_fn((logmean + logmean.T) / 2, np.exp)
    ref_isqrt = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
    iu = np.triu_indices(p, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X_tr = _tangent(F_train, ref_isqrt, iu, coef)
    X_te = _tangent(F_test, ref_isqrt, iu, coef)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
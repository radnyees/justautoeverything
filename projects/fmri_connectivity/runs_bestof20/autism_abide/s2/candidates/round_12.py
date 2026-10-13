"""Tangent-space connectivity on CC200 (no GSR): Ledoit-Wolf covariances of z-scored ROI series,
whitened by the (label-free) group geometric mean, matrix-log, upper triangle; L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps the matrix PD
    x = x / sd
    c, _ = ledoit_wolf(x, assume_centered=True)
    n = c.shape[0]
    c = c + 1e-6 * np.trace(c) / n * np.eye(n)
    return c


def _eig_fun(c, fun):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-10)
    return (v * fun(w)) @ v.T


def _geometric_mean(covs, max_iter=30, tol=1e-6):
    try:
        from pyriemann.utils.mean import mean_riemann
        return mean_riemann(covs, tol=tol, maxiter=max_iter)
    except Exception:
        # fallback: log-Euclidean mean
        logs = np.stack([_eig_fun(c, np.log) for c in covs])
        return _eig_fun(logs.mean(axis=0), np.exp)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_cov(x) for x in ts])
    ref = _geometric_mean(covs)
    isqrt = _eig_fun(ref, lambda w: 1.0 / np.sqrt(w))
    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        w = isqrt @ c @ isqrt
        w = (w + w.T) / 2.0
        t = _eig_fun(w, np.log)
        feats[i] = (t[iu] * coef).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
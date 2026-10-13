"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)) @ V.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        x = x / sd
        c, _ = ledoit_wolf(x, assume_centered=True)
        covs.append(c)
    covs = np.asarray(covs)

    # Label-free reference point: log-Euclidean mean of all covariances.
    logs = np.asarray([_eig_fn(c, np.log) for c in covs])
    ref = _eig_fn(logs.mean(axis=0), np.exp)
    ref_isqrt = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))

    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    for c in covs:
        m = ref_isqrt @ c @ ref_isqrt
        m = (m + m.T) / 2.0
        t = _eig_fn(m, np.log)
        feats.append(t[iu] * coef)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
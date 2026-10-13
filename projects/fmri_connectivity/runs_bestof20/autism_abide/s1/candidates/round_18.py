"""Tangent-space connectivity on CC200 (no GSR): Ledoit-Wolf covariances, log-Euclidean reference, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _spd_fun(c, fun):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-8)
    return (v * fun(w)) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)
    p = covs.shape[1]

    # Label-free reference: log-Euclidean mean of all covariances
    logs_mean = np.mean([_spd_fun(c, np.log) for c in covs], axis=0)
    ref = _spd_fun(logs_mean, np.exp)
    ref_isqrt = _spd_fun(ref, lambda w: 1.0 / np.sqrt(w))

    iu = np.triu_indices(p, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    for c in covs:
        t = _spd_fun(ref_isqrt @ c @ ref_isqrt, np.log)
        feats.append(t[iu] * weights)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
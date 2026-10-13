"""Tangent-space connectivity (Ledoit-Wolf covariances, log-Euclidean reference) on CC200 (no GSR),
upper triangle, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)
    p = covs.shape[1]

    # log-Euclidean group mean (label-free reference point)
    logs = np.asarray([_sym_fn(c, np.log) for c in covs])
    mean = _sym_fn(logs.mean(0), np.exp)
    w_isqrt = _sym_fn(mean, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-10)))

    iu = np.triu_indices(p, k=0)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        t = _sym_fn(w_isqrt @ c @ w_isqrt, lambda w: np.log(np.maximum(w, 1e-10)))
        feats[i] = t[iu] * scale
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
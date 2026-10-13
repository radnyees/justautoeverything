"""Tangent-space embedding of Ledoit-Wolf correlation matrices on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    x = x / sd
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
    d = np.sqrt(np.clip(np.diag(c), 1e-8, None))
    c = c / np.outer(d, d)
    # small ridge for numerical stability (all-zero ROIs)
    c = c + 1e-3 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_lw_corr(x) for x in ts]
    n = covs[0].shape[0]
    # log-Euclidean mean as reference point (label-free)
    log_mean = np.zeros((n, n))
    for c in covs:
        log_mean += _sym_fn(c, lambda w: np.log(np.clip(w, 1e-10, None)))
    log_mean /= len(covs)
    ref = _sym_fn(log_mean, np.exp)
    ref_isqrt = _sym_fn(ref, lambda w: 1.0 / np.sqrt(np.clip(w, 1e-10, None)))
    iu = np.triu_indices(n, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        w = ref_isqrt @ c @ ref_isqrt
        w = (w + w.T) / 2
        t = _sym_fn(w, lambda e: np.log(np.clip(e, 1e-10, None)))
        feats[i] = (t[iu] * weights).astype(np.float32)
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
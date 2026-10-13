"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (athena), L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    bad = sd < 1e-8
    sd[bad] = 1.0
    x = x / sd
    x[:, bad] = 0.0
    c, _ = ledoit_wolf(x, assume_centered=True)
    # make zero-signal ROIs neutral (identity) so they do not create singular directions
    c[bad, :] = 0.0
    c[:, bad] = 0.0
    c[bad, bad] = 1.0
    c = 0.5 * (c + c.T)
    return c


def _eig_fn(c, fn):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-10)
    return (v * fn(w)) @ v.T


def _riemann_mean(covs, n_iter=30, tol=1e-7):
    # start from log-Euclidean mean
    m = _eig_fn(np.mean([_eig_fn(c, np.log) for c in covs], axis=0), np.exp)
    for _ in range(n_iter):
        ms = _eig_fn(m, np.sqrt)
        mis = _eig_fn(m, lambda w: 1.0 / np.sqrt(w))
        t = np.mean([_eig_fn(mis @ c @ mis, np.log) for c in covs], axis=0)
        m = ms @ _eig_fn(t, np.exp) @ ms
        m = 0.5 * (m + m.T)
        if np.linalg.norm(t) < tol:
            break
    return m


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_corr(x) for x in ts])
    ref = _riemann_mean(covs)
    mis = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
    p = covs.shape[1]
    iu = np.triu_indices(p, k=1)
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        t = _eig_fn(mis @ c @ mis, np.log)
        feats[i] = np.sqrt(2.0) * t[iu]
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
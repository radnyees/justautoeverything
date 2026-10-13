"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _clean(x, rng):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    bad = sd < 1e-8
    sd[bad] = 1.0
    x = x / sd
    if bad.any():
        # ROIs outside field of view: tiny independent noise keeps covariance SPD
        x[:, bad] = 1e-3 * rng.standard_normal((x.shape[0], int(bad.sum())))
    return x


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def _geometric_mean(covs, n_iter=30, tol=1e-7):
    # Riemannian (affine-invariant) mean, initialised at the Euclidean mean
    g = np.mean(covs, axis=0)
    for _ in range(n_iter):
        g_isqrt = _sym_fn(g, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))
        g_sqrt = _sym_fn(g, lambda w: np.sqrt(np.maximum(w, 1e-12)))
        t = np.zeros_like(g)
        for c in covs:
            t += _sym_fn(g_isqrt @ c @ g_isqrt, lambda w: np.log(np.maximum(w, 1e-12)))
        t /= len(covs)
        g = g_sqrt @ _sym_fn(t, np.exp) @ g_sqrt
        g = 0.5 * (g + g.T)
        if np.linalg.norm(t) < tol:
            break
    return g


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    rng = np.random.RandomState(0)
    covs = []
    for x in ts:
        xc = _clean(x, rng)
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(xc).covariance_
        covs.append(0.5 * (c + c.T))
    covs = np.asarray(covs)
    g = _geometric_mean(covs)
    g_isqrt = _sym_fn(g, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))
    n = covs.shape[1]
    iu = np.triu_indices(n, k=1)
    feats = []
    for c in covs:
        t = _sym_fn(g_isqrt @ c @ g_isqrt, lambda w: np.log(np.maximum(w, 1e-12)))
        feats.append(np.sqrt(2.0) * t[iu])
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
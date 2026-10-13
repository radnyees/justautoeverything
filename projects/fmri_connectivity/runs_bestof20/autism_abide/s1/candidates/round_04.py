"""Tangent-space embedding of Ledoit-Wolf correlation matrices on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    good = sd > 1e-8
    x[:, good] /= sd[good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # zero-signal ROIs: set to unit variance, no coupling
    bad = np.where(~good)[0]
    c[bad, :] = 0.0
    c[:, bad] = 0.0
    c[bad, bad] = 1.0
    d = np.sqrt(np.clip(np.diag(c), 1e-10, None))
    c = c / np.outer(d, d)
    c = 0.5 * (c + c.T)
    # ensure positive definiteness
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-6, None)
    return (v * w) @ v.T


def _fun_sym(c, fun):
    w, v = np.linalg.eigh(c)
    return (v * fun(w)) @ v.T


def _ref_mean(covs):
    try:
        from pyriemann.utils.mean import mean_riemann
        return mean_riemann(covs, maxiter=50)
    except Exception:
        logs = np.mean([_fun_sym(c, np.log) for c in covs], axis=0)
        return _fun_sym(logs, np.exp)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.asarray([_lw_corr(x) for x in ts])
    ref = _ref_mean(covs)
    w_isqrt = _fun_sym(ref, lambda w: 1.0 / np.sqrt(np.clip(w, 1e-10, None)))
    p = covs.shape[1]
    iu = np.triu_indices(p, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        m = w_isqrt @ c @ w_isqrt
        m = 0.5 * (m + m.T)
        t = _fun_sym(m, lambda w: np.log(np.clip(w, 1e-10, None)))
        feats[i] = (t[iu] * coef).astype(np.float32)
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
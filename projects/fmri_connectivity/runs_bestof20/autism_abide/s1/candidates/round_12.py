"""Tangent-space embedding of Ledoit-Wolf CC200 connectivity (no GSR), reference fit on train fold, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    dead = sd < 1e-8
    sd[dead] = 1.0
    z = x / sd
    z[:, dead] = 0.0
    c = LedoitWolf(assume_centered=True).fit(z).covariance_
    c[dead, :] = 0.0
    c[:, dead] = 0.0
    c[dead, dead] = 1.0
    # normalise to correlation and ensure positive definiteness
    d = np.sqrt(np.clip(np.diag(c), 1e-8, None))
    c = c / np.outer(d, d)
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-6, None)
    return (v * w) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_corr(x) for x in ts]).astype(np.float32)
    return covs


def _fun_sym(c, fn):
    w, v = np.linalg.eigh(c)
    return (v * fn(w)) @ v.T


def _tangent(covs, ref_isqrt, iu, weights):
    out = np.empty((len(covs), len(iu[0])), dtype=np.float64)
    for i, c in enumerate(covs):
        m = ref_isqrt @ np.asarray(c, dtype=np.float64) @ ref_isqrt
        m = (m + m.T) / 2
        t = _fun_sym(m, lambda w: np.log(np.clip(w, 1e-10, None)))
        out[i] = t[iu] * weights
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    n_roi = F_train.shape[1]
    # log-Euclidean mean of training covariances as tangent reference
    logs = np.zeros((n_roi, n_roi))
    for c in F_train:
        logs += _fun_sym(np.asarray(c, dtype=np.float64), lambda w: np.log(np.clip(w, 1e-10, None)))
    logs /= len(F_train)
    ref = _fun_sym((logs + logs.T) / 2, np.exp)
    ref_isqrt = _fun_sym(ref, lambda w: 1.0 / np.sqrt(np.clip(w, 1e-10, None)))

    iu = np.triu_indices(n_roi, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X_tr = _tangent(F_train, ref_isqrt, iu, weights)
    X_te = _tangent(F_test, ref_isqrt, iu, weights)

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
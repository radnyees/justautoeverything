"""Tangent-space embedding of Ledoit-Wolf shrunk correlation matrices (CC200, no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    good = sd > 1e-8
    z = np.zeros_like(x)
    z[:, good] = x[:, good] / sd[good]
    c = LedoitWolf(assume_centered=True).fit(z).covariance_
    # renormalize to correlation scale; dead ROIs get unit diagonal, zero coupling
    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    c = c / np.outer(d, d)
    c[~good, :] = 0.0
    c[:, ~good] = 0.0
    c[np.diag_indices_from(c)] = 1.0
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x).astype(np.float32).ravel() for x in ts]
    return np.asarray(covs, dtype=np.float32)


def _eig_fun(c, fun):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * fun(w)) @ v.T


def _tangent(F, ref_isqrt, iu, coef):
    out = np.empty((F.shape[0], len(iu[0])), dtype=np.float32)
    n = ref_isqrt.shape[0]
    for i in range(F.shape[0]):
        c = F[i].astype(np.float64).reshape(n, n)
        c = 0.5 * (c + c.T)
        w = ref_isqrt @ c @ ref_isqrt
        w = 0.5 * (w + w.T)
        L = _eig_fun(w, np.log)
        out[i] = (L[iu] * coef).astype(np.float32)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    n = int(round(np.sqrt(F_train.shape[1])))
    # log-Euclidean mean of training matrices as reference point
    logsum = np.zeros((n, n))
    for i in range(F_train.shape[0]):
        c = F_train[i].astype(np.float64).reshape(n, n)
        logsum += _eig_fun(0.5 * (c + c.T), np.log)
    ref = _eig_fun(logsum / F_train.shape[0], np.exp)
    ref_isqrt = _eig_fun(ref, lambda w: 1.0 / np.sqrt(w))

    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    T_train = _tangent(F_train, ref_isqrt, iu, coef)
    T_test = _tangent(F_test, ref_isqrt, iu, coef)

    sc = StandardScaler().fit(T_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(T_train), y_train)
    return clf.predict_proba(sc.transform(T_test))[:, 1]
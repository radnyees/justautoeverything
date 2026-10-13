"""Tangent-space connectivity: Ledoit-Wolf shrunk correlation on CC200 (no GSR), whitened by the
training-fold log-Euclidean mean, matrix-log, vectorized; L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    x[:, good] /= sd[good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # renormalize to a correlation matrix; dead ROIs get unit variance and no coupling
    d = np.sqrt(np.clip(np.diag(c), 1e-8, None))
    c = c / np.outer(d, d)
    c[~good, :] = 0.0
    c[:, ~good] = 0.0
    c[np.diag_indices_from(c)] = 1.0
    c += 1e-6 * np.eye(c.shape[0])
    return c.astype(np.float32)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    return np.stack([_shrunk_corr(x) for x in ts]).astype(np.float32)


def _logm(c):
    w, v = np.linalg.eigh(c)
    w = np.log(np.clip(w, 1e-10, None))
    return (v * w) @ v.T


def _inv_sqrtm(c):
    w, v = np.linalg.eigh(c)
    w = 1.0 / np.sqrt(np.clip(w, 1e-10, None))
    return (v * w) @ v.T


def _tangent(C, ref_isqrt, iu, coef):
    out = np.empty((len(C), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(C):
        m = ref_isqrt @ c.astype(np.float64) @ ref_isqrt
        m = (m + m.T) / 2.0
        out[i] = _logm(m)[iu] * coef
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    n = F_train.shape[1]
    # log-Euclidean mean of training matrices as tangent-space reference
    L = np.zeros((n, n))
    for c in F_train:
        L += _logm(c.astype(np.float64))
    L /= len(F_train)
    w, v = np.linalg.eigh(L)
    ref = (v * np.exp(w)) @ v.T
    ref_isqrt = _inv_sqrtm(ref)

    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X_tr = _tangent(F_train, ref_isqrt, iu, coef)
    X_te = _tangent(F_test, ref_isqrt, iu, coef)

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
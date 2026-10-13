"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-3 * np.eye(c.shape[0])  # guarantee SPD (zero ROIs)
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)) @ V.T


def _logeuclid_mean(covs):
    acc = np.zeros(covs[0].shape, dtype=np.float64)
    for c in covs:
        acc += _eig_fn(c.astype(np.float64), np.log)
    acc /= len(covs)
    acc = (acc + acc.T) / 2
    return _eig_fn(acc, np.exp)


def _tangent(covs, ref_isqrt, iu, wts):
    out = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        m = ref_isqrt @ c.astype(np.float64) @ ref_isqrt
        m = (m + m.T) / 2
        L = _eig_fn(m, np.log)
        out[i] = (L[iu] * wts).astype(np.float32)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    ref = _logeuclid_mean(F_train)
    ref_isqrt = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
    p = ref.shape[0]
    iu = np.triu_indices(p, k=0)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    T_train = _tangent(F_train, ref_isqrt, iu, wts)
    T_test = _tangent(F_test, ref_isqrt, iu, wts)
    sc = StandardScaler().fit(T_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(T_train), y_train)
    return clf.predict_proba(sc.transform(T_test))[:, 1]
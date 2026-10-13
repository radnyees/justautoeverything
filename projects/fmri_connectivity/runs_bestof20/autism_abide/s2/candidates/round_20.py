"""CC200 (no GSR) Ledoit-Wolf covariance -> tangent-space embedding at train-fold log-Euclidean mean -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _eig_fun(C, fun):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * fun(w)) @ V.T


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
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _tangent(covs, ref_isqrt):
    n_roi = covs.shape[1]
    iu = np.triu_indices(n_roi, k=0)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((covs.shape[0], len(iu[0])), dtype=np.float32)
    for i, C in enumerate(covs):
        W = ref_isqrt @ C.astype(np.float64) @ ref_isqrt
        W = 0.5 * (W + W.T)
        L = _eig_fun(W, np.log)
        out[i] = L[iu] * wts
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    # log-Euclidean mean of training covariances as reference point
    log_mean = np.mean([_eig_fun(C, np.log) for C in F_train], axis=0)
    ref = _eig_fun(0.5 * (log_mean + log_mean.T), np.exp)
    ref_isqrt = _eig_fun(ref, lambda w: 1.0 / np.sqrt(w))
    X_tr = _tangent(F_train, ref_isqrt)
    X_te = _tangent(F_test, ref_isqrt)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
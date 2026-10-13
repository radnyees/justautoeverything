"""Tangent-space embedding of Ledoit-Wolf shrunk correlation (CC200), L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage gives them positive diagonal
        x = x / sd
        c, _ = ledoit_wolf(x, assume_centered=True)
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _funm(c, fn):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-10)
    return (v * fn(w)) @ v.T


def _tangent(covs, whiten):
    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        m = whiten @ c.astype(np.float64) @ whiten
        m = (m + m.T) / 2
        out[i] = (_funm(m, np.log)[iu] * wts).astype(np.float32)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    # log-Euclidean mean of training covariances as tangent reference point
    logs = np.mean([_funm(c, np.log) for c in F_train], axis=0)
    ref = _funm((logs + logs.T) / 2, np.exp)
    whiten = _funm(ref, lambda w: 1.0 / np.sqrt(w))
    X_tr = _tangent(F_train, whiten)
    X_te = _tangent(F_test, whiten)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
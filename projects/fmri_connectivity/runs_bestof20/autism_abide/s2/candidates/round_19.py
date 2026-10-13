"""Tangent-space connectivity on CC200 (no GSR): Ledoit-Wolf shrunk correlation matrices,
log-Euclidean reference mean, whitened matrix log, upper triangle, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, 1e-10)
    return (v * fn(w)) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps the matrix PD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        covs.append(c)
    covs = np.asarray(covs)

    # Label-free reference: log-Euclidean mean over all items
    logs = np.asarray([_sym_fn(c, np.log) for c in covs])
    ref = _sym_fn(logs.mean(axis=0), np.exp)
    whiten = _sym_fn(ref, lambda w: 1.0 / np.sqrt(w))

    p = covs.shape[1]
    iu = np.triu_indices(p, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    for c in covs:
        t = _sym_fn(whiten @ c @ whiten, np.log)
        feats.append(t[iu] * coef)
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
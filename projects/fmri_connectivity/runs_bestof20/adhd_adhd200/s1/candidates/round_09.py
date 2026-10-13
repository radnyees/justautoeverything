"""Tangent-space embedding of Ledoit-Wolf covariances on CC200, L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _eig_fn(m, fn):
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
        c, _ = ledoit_wolf(x, assume_centered=True)
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)
    n_roi = covs.shape[1]

    # Label-free reference: log-Euclidean mean of all subjects' covariances
    logs = np.asarray([_eig_fn(c, np.log) for c in covs])
    ref = _eig_fn(logs.mean(axis=0), np.exp)
    ref_isqrt = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))

    iu = np.triu_indices(n_roi, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        t = _eig_fn(ref_isqrt @ c @ ref_isqrt, np.log)
        feats[i] = (t[iu] * weights).astype(np.float32)
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
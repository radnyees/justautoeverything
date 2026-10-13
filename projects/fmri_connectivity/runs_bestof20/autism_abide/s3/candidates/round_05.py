"""Tangent-space connectivity on CC200 (no GSR): Ledoit-Wolf covariances, log-Euclidean reference,
whitened matrix log, upper triangle, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps the matrix SPD
        x = x / sd
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)

    # Label-free reference point: log-Euclidean mean of all subjects' covariances
    floor = lambda w: np.log(np.maximum(w, 1e-10))
    logs = np.asarray([_sym_fn(c, floor) for c in covs])
    ref = _sym_fn(logs.mean(axis=0), np.exp)
    ref_isqrt = _sym_fn(ref, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-10)))

    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        w = ref_isqrt @ c @ ref_isqrt
        w = (w + w.T) / 2.0
        t = _sym_fn(w, floor)
        feats[i] = (t[iu] * weights).astype(np.float32)
    return np.nan_to_num(feats)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
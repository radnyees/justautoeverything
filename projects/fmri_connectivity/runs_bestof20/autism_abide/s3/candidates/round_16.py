"""CC200 (no GSR) -> Ledoit-Wolf covariance per subject -> tangent-space embedding
(log-Euclidean mean reference fitted on training fold) -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.stack(covs)


def _tangent(covs, ref_isqrt):
    p = covs.shape[1]
    iu = np.triu_indices(p, k=0)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((covs.shape[0], len(iu[0])), dtype=np.float32)
    for i, C in enumerate(covs):
        M = ref_isqrt @ C.astype(np.float64) @ ref_isqrt
        M = (M + M.T) / 2
        L = _sym_fn(M, lambda v: np.log(np.maximum(v, 1e-10)))
        out[i] = L[iu] * w
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    # log-Euclidean mean of training covariances as tangent reference
    logs = np.zeros(F_train.shape[1:], dtype=np.float64)
    for C in F_train:
        logs += _sym_fn(C.astype(np.float64), lambda v: np.log(np.maximum(v, 1e-10)))
    logs /= len(F_train)
    ref = _sym_fn(logs, np.exp)
    ref_isqrt = _sym_fn(ref, lambda v: 1.0 / np.sqrt(np.maximum(v, 1e-10)))

    X_tr = _tangent(F_train, ref_isqrt)
    X_te = _tangent(F_test, ref_isqrt)

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
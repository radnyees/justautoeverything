"""Tangent-space embedding of Ledoit-Wolf shrunk CC200 correlation matrices (no GSR) + L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps the matrix SPD
        x = x / sd
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _sym_fn(mats, fn):
    w, v = np.linalg.eigh(mats)
    w = np.maximum(w, 1e-10)
    return (v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def _tangent(covs, ref_isqrt, iu, weights):
    out = []
    for i in range(0, len(covs), 64):
        c = np.asarray(covs[i:i + 64], dtype=np.float64)
        wht = ref_isqrt @ c @ ref_isqrt
        wht = 0.5 * (wht + np.swapaxes(wht, -1, -2))
        lg = _sym_fn(wht, np.log)
        out.append(lg[:, iu[0], iu[1]] * weights)
    return np.vstack(out)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    p = F_train.shape[1]
    # log-Euclidean mean of training covariances as tangent reference
    log_mean = np.zeros((p, p))
    for i in range(0, len(F_train), 64):
        log_mean += _sym_fn(F_train[i:i + 64], np.log).sum(axis=0)
    log_mean /= len(F_train)
    ref = _sym_fn(log_mean[None], np.exp)[0]
    ref_isqrt = _sym_fn(ref[None], lambda w: 1.0 / np.sqrt(w))[0]

    iu = np.triu_indices(p, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X_tr = _tangent(F_train, ref_isqrt, iu, weights)
    X_te = _tangent(F_test, ref_isqrt, iu, weights)

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
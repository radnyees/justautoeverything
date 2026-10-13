"""Tangent-space embedding of Ledoit-Wolf correlation matrices (CC200, no GSR), L2 logistic regression."""
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
        dead = sd < 1e-8
        sd[dead] = 1.0
        x = x / sd
        x[:, dead] = 0.0
        c = ledoit_wolf(x, assume_centered=True)[0]
        c[dead, :] = 0.0
        c[:, dead] = 0.0
        c[dead, dead] = 1.0
        covs.append(c.astype(np.float32))
    return np.stack(covs)


def _spd_fn(M, fn):
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _log_euclid_mean(C, chunk=100):
    acc = None
    for i in range(0, len(C), chunk):
        s = _spd_fn(C[i:i + chunk].astype(np.float64), np.log).sum(0)
        acc = s if acc is None else acc + s
    return _spd_fn(acc / len(C), np.exp)


def _tangent(C, isq, chunk=100):
    p = isq.shape[0]
    iu = np.triu_indices(p)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = []
    for i in range(0, len(C), chunk):
        M = isq @ C[i:i + chunk].astype(np.float64) @ isq
        M = (M + np.swapaxes(M, -1, -2)) / 2.0
        L = _spd_fn(M, np.log)
        out.append((L[:, iu[0], iu[1]] * wts).astype(np.float32))
    return np.concatenate(out)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    ref = _log_euclid_mean(F_train)
    isq = _spd_fn(ref, lambda w: 1.0 / np.sqrt(w))
    T_tr = _tangent(F_train, isq)
    T_te = _tangent(F_test, isq)
    sc = StandardScaler().fit(T_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(T_tr), y_train)
    return clf.predict_proba(sc.transform(T_te))[:, 1]
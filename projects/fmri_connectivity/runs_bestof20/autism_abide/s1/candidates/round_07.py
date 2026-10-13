"""CC200 (no GSR) Ledoit-Wolf correlation -> tangent space at training-fold Riemannian mean -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0
        x = (x - x.mean(axis=0)) / sd
        c = LedoitWolf().fit(x).covariance_
        d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
        c = c / np.outer(d, d)
        np.fill_diagonal(c, 1.0)
        if iu is None:
            iu = np.triu_indices(c.shape[0], k=1)
        feats.append(c[iu])
    return np.asarray(feats, dtype=np.float32)


def _to_mats(F):
    F = np.asarray(F, dtype=np.float64)
    L = F.shape[1]
    p = int(round((1 + np.sqrt(1 + 8 * L)) / 2))
    iu = np.triu_indices(p, k=1)
    M = np.zeros((F.shape[0], p, p))
    M[:, iu[0], iu[1]] = F
    M = M + M.transpose(0, 2, 1)
    M[:, np.arange(p), np.arange(p)] = 1.0
    return M


def _fn(C, f):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * f(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _chunked_log_whitened(C, W, chunk=128):
    out = np.empty_like(C)
    for i in range(0, len(C), chunk):
        out[i:i + chunk] = _fn(W @ C[i:i + chunk] @ W, np.log)
    return out


def _riemann_mean(C, maxiter=15, tol=1e-7):
    p = C.shape[1]
    logsum = np.zeros((p, p))
    for i in range(0, len(C), 128):
        logsum += _fn(C[i:i + 128], np.log).sum(axis=0)
    M = _fn(logsum / len(C), np.exp)  # log-Euclidean init
    for _ in range(maxiter):
        Ms = _fn(M, np.sqrt)
        Mi = _fn(M, lambda w: 1.0 / np.sqrt(w))
        T = np.zeros((p, p))
        for i in range(0, len(C), 128):
            T += _fn(Mi @ C[i:i + 128] @ Mi, np.log).sum(axis=0)
        T /= len(C)
        M = Ms @ _fn(T, np.exp) @ Ms
        M = (M + M.T) / 2
        if np.linalg.norm(T) < tol:
            break
    return M


def _tangent(C, M):
    W = _fn(M, lambda w: 1.0 / np.sqrt(w))
    T = _chunked_log_whitened(C, W)
    p = C.shape[1]
    iu = np.triu_indices(p, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return T[:, iu[0], iu[1]] * coef


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = _to_mats(F_train)
    Cte = _to_mats(F_test)
    M = _riemann_mean(Ctr)
    Xtr = _tangent(Ctr, M)
    Xte = _tangent(Cte, M)
    del Ctr, Cte
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
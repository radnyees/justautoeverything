"""Ledoit-Wolf covariance on CC200 (no GSR) -> tangent-space embedding at the training-fold
Riemannian mean -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        x = np.where(sd > 1e-8, x / np.where(sd > 1e-8, sd, 1.0), 0.0)  # all-zero ROIs stay zero
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
        c = c + 1e-6 * np.eye(c.shape[0])  # guarantee positive definiteness
        covs.append(c.astype(np.float32))
    return np.stack(covs)


def _apply(C, fn, chunk=128):
    """Apply a scalar function to the eigenvalues of a stack of SPD matrices."""
    out = np.empty_like(C)
    for s in range(0, len(C), chunk):
        w, V = np.linalg.eigh(C[s:s + chunk])
        w = np.maximum(w, 1e-10)
        out[s:s + chunk] = (V * fn(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    return out


def _sym_fn(M, fn):
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-10)
    return (V * fn(w)) @ V.T


def _riemann_mean(C, n_iter=3):
    # log-Euclidean initialization, then a few Karcher (gradient) steps
    G = _sym_fn(_apply(C, np.log).mean(axis=0), np.exp)
    for _ in range(n_iter):
        Gs = _sym_fn(G, np.sqrt)
        Gis = _sym_fn(G, lambda w: 1.0 / np.sqrt(w))
        T = _apply(Gis @ C @ Gis, np.log).mean(axis=0)
        G = Gs @ _sym_fn(T, np.exp) @ Gs
        G = (G + G.T) / 2
    return G


def _tangent(C, G):
    Gis = _sym_fn(G, lambda w: 1.0 / np.sqrt(w))
    L = _apply(Gis @ C @ Gis, np.log)
    p = C.shape[1]
    iu = np.triu_indices(p, k=0)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return (L[:, iu[0], iu[1]] * w).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    G = _riemann_mean(Ctr)
    Xtr, Xte = _tangent(Ctr, G), _tangent(Cte, G)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
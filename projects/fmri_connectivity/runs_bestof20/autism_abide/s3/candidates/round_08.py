"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
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
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        x[:, good] /= sd[good]
        x[:, ~good] = 0.0
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # dead ROIs: unit variance, no coupling (keeps matrix SPD)
        idx = np.where(~good)[0]
        c[idx, :] = 0.0
        c[:, idx] = 0.0
        c[idx, idx] = 1.0
        c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _powm(C, p):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * w ** p) @ V.T


def _logm(C):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * np.log(w)) @ V.T


def _geometric_mean(covs, n_iter=20, tol=1e-7):
    # Riemannian (affine-invariant) mean, initialised at log-Euclidean mean
    M = np.mean([_logm(c) for c in covs], axis=0)
    w, V = np.linalg.eigh(M)
    G = (V * np.exp(w)) @ V.T
    for _ in range(n_iter):
        Gh = _powm(G, 0.5)
        Gih = _powm(G, -0.5)
        T = np.mean([_logm(Gih @ c @ Gih) for c in covs], axis=0)
        w, V = np.linalg.eigh(T)
        G = Gh @ ((V * np.exp(w)) @ V.T) @ Gh
        G = 0.5 * (G + G.T)
        if np.linalg.norm(T) < tol:
            break
    return G


def _tangent(covs, G):
    Gih = _powm(G, -0.5)
    n = G.shape[0]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        L = _logm(Gih @ c @ Gih)
        out[i] = coef * L[iu]
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    G = _geometric_mean(Ctr)
    Xtr = _tangent(Ctr, G)
    Xte = _tangent(Cte, G)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
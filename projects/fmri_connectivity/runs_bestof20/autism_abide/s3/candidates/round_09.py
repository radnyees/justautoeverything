"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def _riemann_mean(covs, n_iter=20, tol=1e-7):
    # Initialise with the log-Euclidean mean, then do Karcher-mean gradient steps
    M = _sym_fn(np.mean([_sym_fn(c, np.log) for c in covs], axis=0), np.exp)
    for _ in range(n_iter):
        Mh = _sym_fn(M, np.sqrt)
        Mih = _sym_fn(M, lambda w: 1.0 / np.sqrt(w))
        T = np.mean([_sym_fn(Mih @ c @ Mih, np.log) for c in covs], axis=0)
        M = Mh @ _sym_fn(T, np.exp) @ Mh
        if np.linalg.norm(T) < tol:
            break
    return M


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0)
        sd = x.std(0)
        x = x / np.where(sd > 1e-8, sd, 1.0)  # zero (out-of-FOV) ROIs stay zero
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)

    # Label-free reference: geometric mean of all subjects' covariances
    M = _riemann_mean(covs)
    Mih = _sym_fn(M, lambda w: 1.0 / np.sqrt(w))
    p = covs.shape[1]
    iu = np.triu_indices(p, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    for c in covs:
        T = _sym_fn(Mih @ c @ Mih, np.log)
        feats.append(T[iu] * coef)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
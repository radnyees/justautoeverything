"""CC200 (no GSR): Ledoit-Wolf covariance of z-scored ROI series -> per-site Riemannian
re-centering (whiten by each site's Riemannian mean, label-free harmonization) -> tangent
space at the identity -> per-site feature scale equalization (shrunk toward pooled SD)
-> standardized L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.utils.mean import mean_riemann


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        dead = sd < 1e-8
        x = (x - x.mean(axis=0)) / np.where(dead, 1.0, sd)
        x[:, dead] = 0.0
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # Out-of-FOV ROIs: decouple and give unit variance so the matrix stays SPD
        c[dead, :] = 0.0
        c[:, dead] = 0.0
        c[dead, dead] = 1.0
        c = 0.5 * (c + c.T)
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def _sites(meta, n):
    try:
        s = pd.DataFrame(meta)["site"].astype(str).values
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.array(["all"] * n)


def _powm(c, power):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * w ** power) @ v.T


def _log_vec(c, iu, coef):
    w, v = np.linalg.eigh(0.5 * (c + c.T))
    w = np.clip(w, 1e-10, None)
    L = (v * np.log(w)) @ v.T
    return L[iu] * coef


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    ntr, p = Ctr.shape[0], Ctr.shape[1]
    eps = 1e-6 * np.eye(p)
    C = np.concatenate([Ctr, Cte], axis=0) + eps
    sites = np.concatenate([_sites(ctx.get("meta_train"), ntr),
                            _sites(ctx.get("meta_test"), Cte.shape[0])])

    iu = np.triu_indices(p)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X = np.empty((C.shape[0], len(iu[0])), dtype=np.float64)
    # Per-site re-centering: parallel-transport each site's mean to the identity
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        M = mean_riemann(C[idx], maxiter=50) if len(idx) > 1 else C[idx[0]]
        W = _powm(M, -0.5)
        for i in idx:
            X[i] = _log_vec(W @ C[i] @ W, iu, coef)

    # Per-site scale equalization (label-free): z-score each feature within site,
    # with the site variance shrunk toward the pooled variance for small sites.
    pooled_var = X.var(axis=0) + 1e-12
    k = 10.0
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        n_s = len(idx)
        mu = X[idx].mean(axis=0)
        var_s = X[idx].var(axis=0) if n_s > 1 else pooled_var
        v = (n_s * var_s + k * pooled_var) / (n_s + k)
        X[idx] = (X[idx] - mu) / np.sqrt(v + 1e-12)

    Xtr, Xte = X[:ntr], X[ntr:]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
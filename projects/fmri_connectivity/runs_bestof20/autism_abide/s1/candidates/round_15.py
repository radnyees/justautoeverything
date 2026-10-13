"""Ledoit-Wolf covariance on CC200 (no GSR) -> Riemannian tangent space at group geometric mean, L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, logm


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
    x = x / sd
    c, _ = ledoit_wolf(x, assume_centered=True)
    c = 0.5 * (c + c.T)
    c += 1e-3 * np.eye(c.shape[0])  # guarantee SPD (dead ROIs, short scans)
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_cov(x) for x in ts])
    # label-free reference point: geometric (Riemannian) mean of all subjects
    ref = mean_riemann(covs, tol=1e-6, maxiter=50)
    w = invsqrtm(ref)
    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    coeffs = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = np.empty((len(covs), len(iu[0])), dtype=np.float32)
    for i, c in enumerate(covs):
        t = logm(w @ c @ w)
        feats[i] = (coeffs * t[iu]).astype(np.float32)
    return feats


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
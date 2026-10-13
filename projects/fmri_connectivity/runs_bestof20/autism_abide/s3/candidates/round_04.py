"""Tangent-space embedding of Ledoit-Wolf shrunk CC200 correlation matrices (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    bad = sd < 1e-8
    sd[bad] = 1.0
    z = x / sd
    z[:, bad] = 0.0
    c = LedoitWolf(assume_centered=True).fit(z).covariance_
    # renormalize to a correlation-like matrix
    d = np.sqrt(np.clip(np.diag(c), 1e-8, None))
    c = c / np.outer(d, d)
    # ROIs outside the field of view: decouple them (identity rows/cols)
    if bad.any():
        c[bad, :] = 0.0
        c[:, bad] = 0.0
        c[bad, bad] = 1.0
    c = 0.5 * (c + c.T)
    c += 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_corr(x) for x in ts])
    # label-free reference: Riemannian mean over all subjects
    T = TangentSpace(metric="riemann").fit_transform(covs)
    return np.nan_to_num(T).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _subject_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    z = np.zeros_like(x)
    z[:, good] = x[:, good] / sd[good]
    c = LedoitWolf(assume_centered=True).fit(z).covariance_
    # dead (out-of-FOV) ROIs: give unit variance, no coupling
    bad = ~good
    if bad.any():
        c[bad, :] = 0.0
        c[:, bad] = 0.0
        c[bad, bad] = 1.0
    c = 0.5 * (c + c.T) + 1e-4 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_subject_cov(x) for x in ts])
    # label-free reference point: Riemannian mean over all items
    ts_map = TangentSpace(metric="riemann").fit(covs)
    F = ts_map.transform(covs)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F.astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
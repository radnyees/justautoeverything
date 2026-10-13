"""Tangent-space connectivity on CC200 (no GSR): LW-shrunk correlation -> Riemannian tangent space, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x, rng):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    bad = ~np.isfinite(sd) | (sd < 1e-8)
    x = np.nan_to_num(x, nan=0.0)
    sd[bad] = 1.0
    x = x / sd
    if bad.any():
        # impute dead ROIs (outside FOV) with independent unit-variance noise -> ~zero coupling
        x[:, bad] = rng.standard_normal((x.shape[0], int(bad.sum())))
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    rng = np.random.default_rng(0)
    mats = [_shrunk_corr(x, rng) for x in ts]
    return np.asarray(mats, dtype=np.float64)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(F_train)
    X_tr = ts.transform(F_train)
    X_te = ts.transform(F_test)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
"""Tangent-space embedding of Ledoit-Wolf correlation matrices on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    dead = sd < 1e-8
    sd[dead] = 1.0
    x = x / sd
    x[:, dead] = 0.0
    cov = LedoitWolf(assume_centered=True).fit(x).covariance_
    # dead ROIs (outside FOV): isolate them with unit variance so the matrix stays SPD
    cov[dead, :] = 0.0
    cov[:, dead] = 0.0
    cov[dead, dead] = 1.0
    d = np.sqrt(np.diag(cov))
    c = cov / np.outer(d, d)
    c = 0.5 * (c + c.T)
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    C_tr = np.asarray(F_train, dtype=np.float64)
    C_te = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="logeuclid").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
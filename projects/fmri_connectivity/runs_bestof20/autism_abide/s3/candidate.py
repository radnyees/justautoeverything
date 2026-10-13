"""Tangent-space embedding of Ledoit-Wolf shrunk correlation matrices on CC200 (no GSR),
reference = Riemannian mean of the training fold, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    good = sd > 1e-8
    x = np.where(good, x / np.where(good, sd, 1.0), 0.0)
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # dead ROIs: give them identity rows so the matrix stays SPD
    d = np.diag(c).copy()
    bad = ~good.ravel() | (d < 1e-6)
    c[bad, :] = 0.0
    c[:, bad] = 0.0
    c[bad, bad] = 1.0
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c.astype(np.float32)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    return np.stack([_shrunk_corr(x) for x in ts]).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
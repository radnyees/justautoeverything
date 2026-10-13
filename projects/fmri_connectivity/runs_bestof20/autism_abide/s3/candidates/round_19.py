"""Tangent-space connectivity on CC200 (no GSR): per-subject Ledoit-Wolf covariance of standardized
time series, projected at the training-fold Riemannian mean, then L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    x[:, good] /= sd[good]
    x[:, ~good] = 0.0
    c, _ = ledoit_wolf(x, assume_centered=True)
    # guarantee positive definiteness (zero ROIs outside field of view)
    d = np.diag(c).copy()
    bad = d < 1e-3
    if bad.any():
        c[bad, :] = 0.0
        c[:, bad] = 0.0
        c[bad, bad] = 1.0
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_cov(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    C_tr = np.asarray(F_train, dtype=np.float64)
    C_te = np.asarray(F_test, dtype=np.float64)
    C_tr = 0.5 * (C_tr + np.transpose(C_tr, (0, 2, 1)))
    C_te = 0.5 * (C_te + np.transpose(C_te, (0, 2, 1)))
    ts = TangentSpace(metric="riemann").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
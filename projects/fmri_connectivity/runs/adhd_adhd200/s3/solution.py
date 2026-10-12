"""Tangent-space embedding of Ledoit-Wolf covariances on CC200, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    good = sd[0] > 1e-8
    x[:, good] /= sd[:, good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # dead ROIs (outside FOV): give them unit variance, no coupling
    idx = np.where(~good)[0]
    c[idx, :] = 0.0
    c[:, idx] = 0.0
    c[idx, idx] = 1.0
    c += 1e-3 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    C_tr = np.asarray(F_train, dtype=np.float64)
    C_te = np.asarray(F_test, dtype=np.float64)
    # symmetrize after float32 round-trip
    C_tr = 0.5 * (C_tr + C_tr.transpose(0, 2, 1))
    C_te = 0.5 * (C_te + C_te.transpose(0, 2, 1))
    ts = TangentSpace(metric="riemann").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
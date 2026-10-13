"""Tangent-space embedding of Ledoit-Wolf-shrunk correlation matrices on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0)
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps the matrix SPD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # ensure strictly SPD
        c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    return np.asarray(covs, dtype=np.float64)


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
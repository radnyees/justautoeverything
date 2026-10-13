"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
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
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage + ridge keep matrix SPD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs)


def fit_predict(F_train, y_train, F_test, ctx):
    ts = TangentSpace(metric="riemann")
    X_tr = ts.fit_transform(np.asarray(F_train, dtype=np.float64))
    X_te = ts.transform(np.asarray(F_test, dtype=np.float64))
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
"""Ledoit-Wolf covariance of z-scored CC200 time series (no GSR) -> Riemannian tangent space
(reference mean fit on training fold only) -> standardized L2 logistic regression."""
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
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps matrix PD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # small ridge for numerical safety
        c = c + 1e-6 * np.eye(c.shape[0])
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


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
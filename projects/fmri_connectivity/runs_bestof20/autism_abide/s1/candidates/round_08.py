"""CC200 (no GSR): Ledoit-Wolf shrinkage covariance of z-scored time series ->
Riemannian tangent-space embedding (reference mean fit on training fold only) ->
L2 logistic regression."""
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
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
        z = (x - x.mean(axis=0)) / sd
        c = LedoitWolf(assume_centered=True).fit(z).covariance_
        c = 0.5 * (c + c.T) + 1e-4 * np.eye(c.shape[0])  # guarantee SPD
        covs.append(c)
    return np.asarray(covs, dtype=np.float64)


def fit_predict(F_train, y_train, F_test, ctx):
    ts = TangentSpace(metric="riemann").fit(F_train)
    X_tr = ts.transform(F_train)
    X_te = ts.transform(F_test)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
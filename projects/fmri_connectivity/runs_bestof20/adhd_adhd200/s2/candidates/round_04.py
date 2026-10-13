"""Ledoit-Wolf covariance on CC200 (standardized ROI series), Riemannian tangent-space embedding
fit on the training fold, L2 logistic regression."""
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
        good = sd > 1e-8
        x[:, good] /= sd[good]
        x[:, ~good] = 0.0
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        p = c.shape[0]
        # guarantee SPD (all-zero ROIs get only shrinkage mass)
        c = c + 1e-3 * np.eye(p)
        covs.append(c.astype(np.float64))
    return np.asarray(covs)


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
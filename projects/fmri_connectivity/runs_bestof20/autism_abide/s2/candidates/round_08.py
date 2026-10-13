"""Tangent-space embedding of Ledoit-Wolf covariance on CC200 (no GSR), L2 logistic regression."""
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
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(sd > 1e-8, sd, 1.0)
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)
    # Label-free reference point: Riemannian mean over all subjects.
    ts_emb = TangentSpace(metric="riemann").fit(covs)
    return ts_emb.transform(covs).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
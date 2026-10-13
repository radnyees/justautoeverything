"""Tangent-space embedding of Ledoit-Wolf shrunk covariances on CC200 (Athena), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero (out-of-FOV) ROIs stay zero
    x = x / sd
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    c = 0.5 * (c + c.T)
    # guarantee strict positive definiteness
    c += 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_shrunk_corr(x) for x in ts])
    # label-free reference point: Riemannian mean of all subjects
    feats = TangentSpace(metric="riemann").fit_transform(covs)
    return np.nan_to_num(feats).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
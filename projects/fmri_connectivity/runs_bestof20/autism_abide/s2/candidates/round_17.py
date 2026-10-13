"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    bad = sd[0] < 1e-8
    sd[:, bad] = 1.0
    x = x / sd
    if bad.any():
        # all-zero ROIs (outside FOV): fill with deterministic small-variance noise so covariance is PD
        rng = np.random.RandomState(0)
        x[:, bad] = rng.randn(x.shape[0], int(bad.sum())) * 0.1
    return x


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        z = _zscore(x)
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(z).covariance_
        # normalize to correlation-like scale
        d = np.sqrt(np.clip(np.diag(c), 1e-8, None))
        c = c / np.outer(d, d)
        c = 0.5 * (c + c.T)
        covs.append(c)
    covs = np.asarray(covs, dtype=np.float64)
    # label-free reference: Riemannian (geometric) mean over all subjects
    feats = TangentSpace(metric="riemann").fit_transform(covs)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
"""CC200 (no GSR) -> Ledoit-Wolf covariance of z-scored ROI series -> Riemannian tangent space
(reference = label-free geometric mean over all items) -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    dead = sd < 1e-8
    sd[dead] = 1.0
    z = x / sd
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(z).covariance_
    if dead.any():
        c[dead, :] = 0.0
        c[:, dead] = 0.0
        c[dead, dead] = 1.0
    c = 0.5 * (c + c.T)
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_cov(x) for x in ts])
    # small ridge for numerical safety in the matrix log
    n = covs.shape[1]
    covs = covs + 1e-6 * np.eye(n)[None]
    ts_feats = TangentSpace(metric="riemann").fit_transform(covs)
    return np.asarray(ts_feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
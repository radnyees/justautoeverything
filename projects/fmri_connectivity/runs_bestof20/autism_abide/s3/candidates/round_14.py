"""CC200 (no GSR) -> Ledoit-Wolf shrunk correlation -> Riemannian tangent space -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    good = sd > 1e-8
    x = np.where(good, x / np.where(good, sd, 1.0), 0.0)
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
    # Zero (out-of-FOV) ROIs: give them a unit diagonal so the matrix stays SPD
    bad = ~good.ravel()
    if bad.any():
        c[bad, :] = 0.0
        c[:, bad] = 0.0
        c[bad, bad] = 1.0
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_shrunk_corr(x) for x in ts])
    # Label-free reference: Riemannian geometric mean over all items
    ts_proj = TangentSpace(metric="riemann").fit_transform(covs)
    return np.asarray(ts_proj, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
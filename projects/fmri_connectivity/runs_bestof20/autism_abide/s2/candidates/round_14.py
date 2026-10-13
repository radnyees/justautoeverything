"""Tangent-space connectivity (Ledoit-Wolf covariances, Riemannian reference mean) on CC200 (no GSR),
L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps the matrix SPD
    x = x / sd
    c = LedoitWolf(store_precision=False).fit(x).covariance_
    c = 0.5 * (c + c.T)
    c[np.diag_indices_from(c)] += 1e-6
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_cov(x) for x in ts])
    # Label-free reference point: Riemannian mean over all subjects.
    tsp = TangentSpace(metric="riemann").fit(covs)
    return np.asarray(tsp.transform(covs), dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
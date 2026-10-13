"""Tangent-space connectivity on CC200 (no GSR): per-subject Ledoit-Wolf correlation,
projected at the Riemannian mean of all subjects (label-free), L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps the matrix PD
    x = x / sd
    c, _ = ledoit_wolf(x, assume_centered=True)
    c = 0.5 * (c + c.T)
    # guard against tiny/zero diagonals
    c += 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_corr(x) for x in ts])
    ts_map = TangentSpace(metric="riemann").fit(covs)
    feats = ts_map.transform(covs)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
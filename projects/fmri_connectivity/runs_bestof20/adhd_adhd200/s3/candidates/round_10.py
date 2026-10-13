"""Ledoit-Wolf shrinkage covariance on CC200 (z-scored ROI signals) -> Riemannian tangent space,
upper triangle, L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    good = sd > 1e-8
    x = np.where(good, x / np.where(good, sd, 1.0), 0.0)  # all-zero ROIs stay zero
    cov, _ = ledoit_wolf(x, assume_centered=True)
    # guarantee SPD (zero ROIs get a small diagonal via shrinkage target, add floor anyway)
    p = cov.shape[0]
    cov = cov + 1e-6 * np.trace(cov) / p * np.eye(p)
    return cov


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_shrunk_corr(x) for x in ts])
    # label-free reference: Riemannian mean of all subjects' covariances
    tsp = TangentSpace(metric="riemann")
    F = tsp.fit_transform(covs)
    return np.nan_to_num(F).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
"""CC200 (athena) Ledoit-Wolf shrunk correlation -> Riemannian tangent-space embedding
(reference mean fitted on the training fold only) -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage gives them a positive diagonal
    x = x / sd
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
    # renormalize to a correlation-like matrix and guarantee SPD
    d = np.sqrt(np.clip(np.diag(c), 1e-6, None))
    c = c / np.outer(d, d)
    c = 0.5 * (c + c.T) + 1e-4 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "athena" if "athena" in ctx["strategies"] else (
        "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0])
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
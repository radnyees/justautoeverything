"""CC200 Ledoit-Wolf covariance -> Riemannian tangent-space embedding, L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero
    x = x / sd
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    p = c.shape[0]
    c = c + 1e-3 * np.eye(p)  # guarantee SPD (zero ROIs, short scans)
    return 0.5 * (c + c.T)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_shrunk_corr(x) for x in ts])
    # Label-free reference point: Riemannian mean of all subjects' covariances.
    ts_emb = TangentSpace(metric="riemann").fit_transform(covs)
    return np.asarray(ts_emb, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
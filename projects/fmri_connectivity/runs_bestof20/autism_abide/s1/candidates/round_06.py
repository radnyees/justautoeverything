"""CC200 (no GSR) Ledoit-Wolf shrunk correlation -> Riemannian tangent space (train-fold reference) -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    x[:, good] /= sd[good]
    x[:, ~good] = 0.0
    cov, _ = ledoit_wolf(x, assume_centered=True)
    d = np.diag(cov).copy()
    d[d <= 1e-10] = 1e-10
    cov[np.diag_indices_from(cov)] = d
    inv = 1.0 / np.sqrt(d)
    c = cov * inv[:, None] * inv[None, :]
    c = 0.5 * (c + c.T)
    # tiny ridge to guarantee positive definiteness (all-zero ROIs, short scans)
    c[np.diag_indices_from(c)] += 1e-6
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x).astype(np.float32) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = np.asarray(F_train, dtype=np.float64)
    Xte = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(Xtr)
    Ttr = ts.transform(Xtr)
    Tte = ts.transform(Xte)
    sc = StandardScaler().fit(Ttr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Ttr), y_train)
    return clf.predict_proba(sc.transform(Tte))[:, 1]
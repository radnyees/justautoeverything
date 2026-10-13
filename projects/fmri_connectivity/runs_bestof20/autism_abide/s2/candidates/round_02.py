"""CC200 (no GSR) -> Ledoit-Wolf shrunk correlation matrices -> Riemannian tangent space
(reference mean fit on training fold only) -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_corr(x, eps=1e-3):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    x[:, good] /= sd[good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # all-zero ROIs (outside FOV): make them isolated unit-variance nodes
    c[~good, :] = 0.0
    c[:, ~good] = 0.0
    c[~good, ~good] = 1.0
    c = c + eps * np.eye(c.shape[0])
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    return 0.5 * (c + c.T)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    mats = [_lw_corr(x) for x in ts]
    return np.asarray(mats, dtype=np.float32)


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
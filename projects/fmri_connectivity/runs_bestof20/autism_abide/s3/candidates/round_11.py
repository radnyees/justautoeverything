"""CC200 (no GSR) Ledoit-Wolf covariances -> tangent-space embedding (reference = Riemannian mean
of training fold only) -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    sd = x.std(axis=0)
    dead = sd < 1e-8
    sd[dead] = 1.0
    z = (x - x.mean(axis=0)) / sd
    z[:, dead] = 0.0
    c = LedoitWolf(assume_centered=True).fit(z).covariance_
    # dead ROIs (outside field of view): make them neutral identity entries so the SPD matrix is well-conditioned
    if dead.any():
        c[dead, :] = 0.0
        c[:, dead] = 0.0
        c[dead, dead] = 1.0
    c = 0.5 * (c + c.T)
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_corr(x) for x in ts]).astype(np.float32)
    return covs


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
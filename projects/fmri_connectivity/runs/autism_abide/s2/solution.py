"""CC200 (no GSR) -> Ledoit-Wolf covariance of standardized signals -> Riemannian tangent space
(reference = geometric mean of training fold) -> standardized L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps matrix PD
    x = x / sd
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
    # guarantee strict positive definiteness
    c = c + 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = np.stack([_lw_cov(x) for x in ts]).astype(np.float32)
    return covs


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    # symmetrize after float32 round trip
    Ctr = 0.5 * (Ctr + np.transpose(Ctr, (0, 2, 1)))
    Cte = 0.5 * (Cte + np.transpose(Cte, (0, 2, 1)))
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
"""CC200 (no GSR) Ledoit-Wolf covariance -> tangent-space embedding (ref = train-fold Riemannian mean), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        x = x / np.where(sd > 1e-8, sd, 1.0)  # all-zero ROIs stay zero; LW shrinkage keeps cov PD
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        if iu is None:
            iu = np.triu_indices(c.shape[0], k=0)
        feats.append(c[iu])
    return np.asarray(feats, dtype=np.float64)


def _to_mats(F):
    F = np.asarray(F, dtype=np.float64)
    m = F.shape[1]
    p = int(round((np.sqrt(8 * m + 1) - 1) / 2))
    iu = np.triu_indices(p, k=0)
    M = np.zeros((F.shape[0], p, p))
    M[:, iu[0], iu[1]] = F
    M = M + np.transpose(M, (0, 2, 1)) - M * np.eye(p)[None]
    # guard against numerical non-PD-ness
    M += 1e-6 * np.eye(p)[None]
    return M


def fit_predict(F_train, y_train, F_test, ctx):
    C_tr = _to_mats(F_train)
    C_te = _to_mats(F_test)
    ts = TangentSpace(metric="riemann").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
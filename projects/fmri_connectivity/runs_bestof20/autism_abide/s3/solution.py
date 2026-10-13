"""Tangent-space embedding of Ledoit-Wolf covariances on CC200 (no GSR), L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    feats = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-8] = np.inf  # all-zero ROIs stay zero; LW shrinkage keeps matrix PD
        x = x / sd
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = 0.5 * (c + c.T)
        feats.append(c.ravel().astype(np.float32))
    return np.asarray(feats, dtype=np.float32)


def _to_mats(F):
    F = np.asarray(F, dtype=np.float64)
    p = int(round(np.sqrt(F.shape[1])))
    M = F.reshape(-1, p, p)
    M = 0.5 * (M + M.transpose(0, 2, 1))
    # tiny ridge for numerical safety
    M = M + 1e-6 * np.eye(p)[None]
    return M


def fit_predict(F_train, y_train, F_test, ctx):
    M_tr = _to_mats(F_train)
    M_te = _to_mats(F_test)
    ts = TangentSpace(metric="riemann").fit(M_tr)
    X_tr = ts.transform(M_tr)
    X_te = ts.transform(M_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
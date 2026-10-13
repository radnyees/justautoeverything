"""Tangent-space embedding of Ledoit-Wolf covariances on CC200, L2 logistic regression."""
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
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; shrinkage keeps cov PD
        x = (x - x.mean(axis=0)) / sd
        cov = LedoitWolf(assume_centered=True).fit(x).covariance_
        if iu is None:
            iu = np.triu_indices(cov.shape[0], k=0)
        feats.append(cov[iu])
    return np.asarray(feats, dtype=np.float32)


def _to_mats(F):
    F = np.asarray(F, dtype=np.float64)
    L = F.shape[1]
    n = int(round((np.sqrt(8 * L + 1) - 1) / 2))
    iu = np.triu_indices(n, k=0)
    M = np.zeros((F.shape[0], n, n))
    M[:, iu[0], iu[1]] = F
    M = M + np.transpose(M, (0, 2, 1))
    M[:, np.arange(n), np.arange(n)] /= 2.0
    # tiny ridge for numerical safety
    M += 1e-6 * np.eye(n)[None]
    return M


def fit_predict(F_train, y_train, F_test, ctx):
    C_tr = _to_mats(F_train)
    C_te = _to_mats(F_test)
    ts = TangentSpace(metric="riemann").fit(C_tr)
    T_tr = ts.transform(C_tr)
    T_te = ts.transform(C_te)
    sc = StandardScaler().fit(T_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(T_tr), y_train)
    return clf.predict_proba(sc.transform(T_te))[:, 1]
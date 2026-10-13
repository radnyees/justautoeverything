PARAMS = {'atlas': 'cc400', 'strategy': 1, 'conn': 'partial', 'site_std': True, 'model': 'lr_l2', 'C': 0.020519502732633327}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def featurize(items, ctx):
    strat = ctx["strategies"][min(PARAMS["strategy"], len(ctx["strategies"]) - 1)]
    ts = ctx["load_timeseries"](items, PARAMS["atlas"], strat)
    covs = []
    for x in ts:
        x = np.nan_to_num(x.astype(np.float64))
        sd = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        covs.append(LedoitWolf().fit(x).covariance_.astype(np.float32))
    return np.stack(covs)


def _vectorize(C, ref=None):
    n = C.shape[1]
    iu = np.triu_indices(n, k=1)
    kind = PARAMS["conn"]
    if kind == "corr":
        d = np.sqrt(np.clip(np.einsum("kii->ki", C), 1e-9, None))
        R = C / d[:, :, None] / d[:, None, :]
        return np.arctanh(np.clip(R[:, iu[0], iu[1]], -0.999, 0.999))
    if kind == "partial":
        P = np.linalg.pinv(C.astype(np.float64))
        d = np.sqrt(np.clip(np.einsum("kii->ki", P), 1e-9, None))
        R = -P / d[:, :, None] / d[:, None, :]
        return np.arctanh(np.clip(R[:, iu[0], iu[1]], -0.999, 0.999))
    # tangent space at the reference (log-Euclidean mean of training covariances)
    w, v = np.linalg.eigh(ref)
    isq = (v / np.sqrt(np.clip(w, 1e-9, None))) @ v.T
    out = []
    for c in C.astype(np.float64):
        m = isq @ c @ isq
        w2, v2 = np.linalg.eigh((m + m.T) / 2)
        L = (v2 * np.log(np.clip(w2, 1e-9, None))) @ v2.T
        out.append(L[iu])
    return np.asarray(out)


def _site_std(X, sites):
    X = X.copy()
    for s in np.unique(sites):
        m = sites == s
        X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0) + 1e-6)
    return X


def fit_predict(F_train, y_train, F_test, ctx):
    ref = None
    if PARAMS["conn"] == "tangent":
        logs = []
        for c in F_train.astype(np.float64):
            w, v = np.linalg.eigh(c)
            logs.append((v * np.log(np.clip(w, 1e-9, None))) @ v.T)
        w, v = np.linalg.eigh(np.mean(logs, axis=0))
        ref = (v * np.exp(w)) @ v.T
    Xtr, Xte = _vectorize(F_train, ref), _vectorize(F_test, ref)
    if PARAMS["site_std"]:
        Xtr = _site_std(Xtr, np.asarray(ctx["meta_train"]["site"]))
        Xte = _site_std(Xte, np.asarray(ctx["meta_test"]["site"]))
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = PARAMS["model"]
    if m == "svm_linear":
        clf = LinearSVC(C=PARAMS["C"] / 100, max_iter=20000).fit(Xtr, y_train)
        return clf.decision_function(Xte)
    if m == "ridge":
        clf = RidgeClassifier(alpha=1.0 / PARAMS["C"]).fit(Xtr, y_train)
        return clf.decision_function(Xte)
    clf = LogisticRegression(C=PARAMS["C"] / 100, max_iter=5000).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]

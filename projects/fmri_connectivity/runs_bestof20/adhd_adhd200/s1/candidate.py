"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(x.T)
        c = np.nan_to_num(c, nan=0.0)
        if iu is None:
            iu = np.triu_indices(c.shape[0], k=1)
        feats.append(np.arctanh(np.clip(c[iu], -0.999, 0.999)))
    return np.asarray(feats, dtype=np.float32)


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        if "site" in meta.columns:
            return meta["site"].astype(str).to_numpy()
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]])
    try:
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(X, sites):
    """Label-free per-site z-scoring (pooled over train+test items of each site)."""
    X = X.astype(np.float64, copy=True)
    for s in np.unique(sites):
        m = sites == s
        if m.sum() < 3:
            X[m] -= X.mean(axis=0, keepdims=True)
            continue
        mu = X[m].mean(axis=0, keepdims=True)
        sd = X[m].std(axis=0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        X[m] = (X[m] - mu) / sd
    return X


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    ntr, nte = len(F_train), len(F_test)
    s_tr = _sites(ctx.get("meta_train"), ntr)
    s_te = _sites(ctx.get("meta_test"), nte)
    X = np.vstack([F_train, F_test])
    S = np.concatenate([s_tr.astype(str), s_te.astype(str)])
    X = _site_standardize(X, S)
    Xtr, Xte = X[:ntr], X[ntr:]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
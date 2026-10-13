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
            return meta["site"].astype(str).values
        return meta.iloc[:, 0].astype(str).values
    if isinstance(meta, dict):
        return np.asarray(meta["site"]).astype(str)
    arr = np.asarray(meta)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.astype(str)


def _site_standardize(X, sites):
    """Location/scale harmonization: z-score every feature within each site (label-free)."""
    X = np.asarray(X, dtype=np.float64).copy()
    for s in np.unique(sites):
        m = sites == s
        if m.sum() < 2:
            continue
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd < 1e-6] = 1.0
        X[m] = (X[m] - mu) / sd
    return X


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = np.asarray(F_train, dtype=np.float64)
    Xte = np.asarray(F_test, dtype=np.float64)
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te])
    X_all = _site_standardize(X_all, s_all)
    Xtr, Xte = X_all[: len(Xtr)], X_all[len(Xtr):]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
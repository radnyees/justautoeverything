"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site feature
standardization (label-free harmonization), L2 logistic regression."""
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
        return meta["site"].astype(str).to_numpy()
    if isinstance(meta, dict):
        return np.asarray([str(s) for s in meta["site"]], dtype=object)
    out = []
    for m in meta:
        out.append(str(m["site"]) if isinstance(m, dict) else str(m))
    return np.asarray(out, dtype=object)


def _site_standardize(X, sites, min_n=5):
    """Center and scale each feature within site; tiny sites only get centered."""
    Xh = np.empty_like(X, dtype=np.float64)
    g_std = X.std(axis=0) + 1e-6
    for s in np.unique(sites):
        m = sites == s
        Xs = X[m].astype(np.float64)
        mu = Xs.mean(axis=0)
        if m.sum() >= min_n:
            sd = Xs.std(axis=0) + 1e-6
        else:
            sd = g_std
        Xh[m] = (Xs - mu) / sd
    return Xh


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    n_tr = F_train.shape[0]
    s_tr = _sites(ctx.get("meta_train"), n_tr)
    s_te = _sites(ctx.get("meta_test"), F_test.shape[0])

    X_all = np.vstack([F_train, F_test])
    s_all = np.concatenate([s_tr, s_te])
    X_all = _site_standardize(X_all, s_all)
    Xtr, Xte = X_all[:n_tr], X_all[n_tr:]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
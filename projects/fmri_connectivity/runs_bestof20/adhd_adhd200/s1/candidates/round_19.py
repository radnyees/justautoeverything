"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site standardization
(label-free harmonization), L2 logistic regression."""
import numpy as np
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
    try:
        import pandas as pd
        if isinstance(meta, pd.DataFrame):
            return meta["site"].astype(str).values
        if isinstance(meta, pd.Series):
            return meta.astype(str).values
    except Exception:
        pass
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]])
    out = []
    for m in meta:
        if isinstance(m, dict):
            out.append(str(m.get("site", "")))
        else:
            out.append(str(m))
    return np.asarray(out)


def _site_standardize(X, sites):
    X = X.astype(np.float64, copy=True)
    for s in np.unique(sites):
        idx = sites == s
        if idx.sum() < 2:
            continue
        mu = X[idx].mean(axis=0)
        sd = X[idx].std(axis=0)
        sd[sd < 1e-6] = 1.0
        X[idx] = (X[idx] - mu) / sd
    return X


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    ntr = len(F_train)
    s_tr = _sites(ctx.get("meta_train"), ntr)
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    # Label-free per-site harmonization using all available (train + test) features of each site
    X_all = np.vstack([F_train, F_test])
    s_all = np.concatenate([s_tr, s_te])
    X_all = _site_standardize(X_all, s_all)
    Xtr, Xte = X_all[:ntr], X_all[ntr:]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
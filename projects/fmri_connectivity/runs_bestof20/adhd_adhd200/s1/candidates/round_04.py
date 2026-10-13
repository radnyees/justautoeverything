"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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
            return meta["site"].astype(str).to_numpy()
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray(meta["site"]).astype(str)
    try:
        return np.asarray([m["site"] if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(X_tr, X_te, s_tr, s_te):
    X_tr = np.asarray(X_tr, dtype=np.float64).copy()
    X_te = np.asarray(X_te, dtype=np.float64).copy()
    # Global fallback stats
    g_mu = X_tr.mean(0)
    g_sd = X_tr.std(0) + 1e-6
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr = s_tr == s
        mte = s_te == s
        pool = np.vstack([X_tr[mtr], X_te[mte]])
        if pool.shape[0] >= 5:
            mu = pool.mean(0)
            sd = pool.std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        X_tr[mtr] = (X_tr[mtr] - mu) / sd
        X_te[mte] = (X_te[mte] - mu) / sd
    return X_tr, X_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    if len(s_tr) != len(F_train) or len(s_te) != len(F_test):
        s_tr = np.zeros(len(F_train), dtype=object)
        s_te = np.zeros(len(F_test), dtype=object)
    Xtr, Xte = _site_standardize(F_train, F_test, s_tr, s_te)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
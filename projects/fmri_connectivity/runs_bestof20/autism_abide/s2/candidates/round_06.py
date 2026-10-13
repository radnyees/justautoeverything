"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site z-scoring
(label-free location/scale harmonization), L2 logistic regression."""
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
            return meta["site"].astype(str).to_numpy()
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]], dtype=object)
    try:
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta], dtype=object)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=3):
    F_train = np.asarray(F_train, dtype=np.float64).copy()
    F_test = np.asarray(F_test, dtype=np.float64).copy()
    F_all = np.vstack([F_train, F_test])
    s_all = np.concatenate([s_train, s_test])
    g_mu = F_all.mean(0)
    g_sd = F_all.std(0) + 1e-6
    for s in np.unique(s_all):
        m = s_all == s
        if m.sum() >= min_n:
            mu = F_all[m].mean(0)
            sd = F_all[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        tr = s_train == s
        te = s_test == s
        F_train[tr] = (F_train[tr] - mu) / sd
        F_test[te] = (F_test[te] - mu) / sd
    return F_train, F_test


def fit_predict(F_train, y_train, F_test, ctx):
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site standardization
(label-free, pooled train+test per site), L2 logistic regression."""
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
        return np.asarray(meta["site"]).astype(str)
    try:
        return np.asarray([m["site"] if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test):
    F_train = np.asarray(F_train, dtype=np.float64).copy()
    F_test = np.asarray(F_test, dtype=np.float64).copy()
    allF = np.vstack([F_train, F_test])
    allS = np.concatenate([s_train, s_test])
    for s in np.unique(allS):
        m = allS == s
        if m.sum() < 3:
            continue
        mu = allF[m].mean(0)
        sd = allF[m].std(0) + 1e-6
        F_train[s_train == s] = (F_train[s_train == s] - mu) / sd
        F_test[s_test == s] = (F_test[s_test == s] - mu) / sd
    return F_train, F_test


def fit_predict(F_train, y_train, F_test, ctx):
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    F_train, F_test = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
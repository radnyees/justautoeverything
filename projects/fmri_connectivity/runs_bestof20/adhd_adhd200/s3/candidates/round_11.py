"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization, L2 logistic regression."""
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
    return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta], dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test):
    """Label-free per-site z-scoring using all subjects (train+test) of each site."""
    F_all = np.vstack([F_train, F_test]).astype(np.float64)
    s_all = np.concatenate([s_train, s_test])
    out = np.empty_like(F_all)
    for s in np.unique(s_all):
        m = s_all == s
        mu = F_all[m].mean(axis=0)
        sd = F_all[m].std(axis=0) if m.sum() > 1 else np.ones(F_all.shape[1])
        sd[sd < 1e-6] = 1.0
        out[m] = (F_all[m] - mu) / sd
    n = len(F_train)
    return out[:n], out[n:]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization, L2 logistic regression."""
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
        if hasattr(meta, "columns") and "site" in meta.columns:
            return np.asarray(meta["site"].values).astype(str)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
        if len(meta) == n and isinstance(meta[0], dict):
            return np.asarray([m.get("site", "") for m in meta]).astype(str)
        arr = np.asarray(meta)
        if arr.ndim == 1 and len(arr) == n:
            return arr.astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object).astype(str)


def _site_standardize(F, sites, eps=1e-6):
    """Label-free per-site z-scoring of every feature."""
    F = np.asarray(F, dtype=np.float64).copy()
    for s in np.unique(sites):
        m = sites == s
        if m.sum() < 2:
            continue
        mu = F[m].mean(axis=0)
        sd = F[m].std(axis=0)
        F[m] = (F[m] - mu) / (sd + eps)
    return F


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Xtr = _site_standardize(F_train, s_tr)
    Xte = _site_standardize(F_test, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
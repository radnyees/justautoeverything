"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
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
        if hasattr(meta, "columns") and "site" in meta.columns:
            return np.asarray(meta["site"]).astype(str)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
        if isinstance(meta, (list, tuple)) and len(meta) and isinstance(meta[0], dict):
            return np.asarray([str(m.get("site", "")) for m in meta])
        arr = np.asarray(meta)
        if arr.ndim == 1 and len(arr) == n:
            return arr.astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _site_standardize(X_tr, X_te, s_tr, s_te):
    """Z-score each feature within site using pooled (label-free) train+test features of that site."""
    X_all = np.vstack([X_tr, X_te]).astype(np.float64)
    s_all = np.concatenate([s_tr, s_te])
    out = np.empty_like(X_all)
    for s in np.unique(s_all):
        m = s_all == s
        mu = X_all[m].mean(axis=0)
        sd = X_all[m].std(axis=0) if m.sum() > 2 else np.ones(X_all.shape[1])
        sd[sd < 1e-6] = 1.0
        out[m] = (X_all[m] - mu) / sd
    return out[: len(X_tr)], out[len(X_tr):]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
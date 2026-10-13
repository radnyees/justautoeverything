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
        return None
    try:
        if hasattr(meta, "columns") and "site" in meta.columns:
            s = np.asarray(meta["site"]).astype(str)
        elif isinstance(meta, dict) and "site" in meta:
            s = np.asarray(meta["site"]).astype(str)
        else:
            s = np.asarray([m["site"] for m in meta]).astype(str)
    except Exception:
        return None
    return s if len(s) == n else None


def _site_standardize(X_tr, X_te, s_tr, s_te):
    """Label-free per-site z-scoring using all (train+test) subjects of each site."""
    X_tr = X_tr.astype(np.float64).copy()
    X_te = X_te.astype(np.float64).copy()
    X_all = np.vstack([X_tr, X_te])
    s_all = np.concatenate([s_tr, s_te])
    g_mu = X_all.mean(0)
    g_sd = X_all.std(0) + 1e-6
    for site in np.unique(s_all):
        m = s_all == site
        if m.sum() >= 3:
            mu = X_all[m].mean(0)
            sd = X_all[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        X_tr[s_tr == site] = (X_tr[s_tr == site] - mu) / sd
        X_te[s_te == site] = (X_te[s_te == site] - mu) / sd
    return X_tr, X_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    if s_tr is not None and s_te is not None:
        F_train, F_test = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
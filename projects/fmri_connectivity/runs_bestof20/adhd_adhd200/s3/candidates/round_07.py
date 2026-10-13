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
        return meta["site"].astype(str).to_numpy() if "site" in meta.columns else np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]], dtype=object)
    try:
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta], dtype=object)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(X_tr, X_te, s_tr, s_te, min_n=5):
    X_tr = X_tr.astype(np.float64).copy()
    X_te = X_te.astype(np.float64).copy()
    stats = {}
    for s in np.unique(s_tr):
        m = s_tr == s
        if m.sum() >= min_n:
            mu = X_tr[m].mean(0)
            sd = X_tr[m].std(0) + 1e-6
            stats[s] = (mu, sd)
    g_mu, g_sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    for s in np.unique(s_tr):
        m = s_tr == s
        mu, sd = stats.get(s, (g_mu, g_sd))
        X_tr[m] = (X_tr[m] - mu) / sd
    for s in np.unique(s_te):
        m = s_te == s
        if s in stats:
            mu, sd = stats[s]
        elif m.sum() >= min_n:
            # unseen site: label-free standardization with its own statistics
            mu, sd = X_te[m].mean(0), X_te[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        X_te[m] = (X_te[m] - mu) / sd
    return X_tr, X_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    X_tr, X_te = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
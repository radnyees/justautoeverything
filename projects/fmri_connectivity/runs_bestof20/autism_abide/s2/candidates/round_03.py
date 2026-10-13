"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site standardization
(label-free location/scale harmonization), L2 logistic regression."""
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
        s = meta["site"]
        s = s.values if hasattr(s, "values") else s
        s = np.asarray(s, dtype=object)
        if len(s) == n:
            return s
    except Exception:
        pass
    try:
        s = np.asarray([m["site"] for m in meta], dtype=object)
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=3):
    """Z-score each feature within site, using pooled (label-free) train+test features of that site."""
    F_all = np.vstack([F_train, F_test]).astype(np.float64)
    s_all = np.concatenate([s_train, s_test])
    g_mu = F_all.mean(0)
    g_sd = F_all.std(0) + 1e-6
    out = (F_all - g_mu) / g_sd
    for site in np.unique(s_all):
        m = s_all == site
        if m.sum() < min_n:
            continue
        mu = F_all[m].mean(0)
        sd = F_all[m].std(0) + 1e-6
        out[m] = (F_all[m] - mu) / sd
    n_tr = len(F_train)
    return out[:n_tr], out[n_tr:]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    X_tr, X_te = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
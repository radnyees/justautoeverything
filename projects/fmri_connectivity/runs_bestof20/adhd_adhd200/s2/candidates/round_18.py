"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
(ComBat-lite location/scale harmonization), L2 logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strats = ctx["strategies"]
    if "athena" in strats:
        strategy = "athena"
    elif "filt_noglobal" in strats:
        strategy = "filt_noglobal"
    else:
        strategy = strats[0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        sd = x.std(axis=0)
        x = np.where(sd > 1e-8, x, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
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
            return np.asarray(meta["site"].astype(str).values)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray([str(s) for s in meta["site"]])
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    Xtr = np.asarray(F_train, dtype=np.float64).copy()
    Xte = np.asarray(F_test, dtype=np.float64).copy()
    g_mu = Xtr.mean(0)
    g_sd = Xtr.std(0) + 1e-6
    stats = {}
    for s in np.unique(s_train):
        m = s_train == s
        if m.sum() >= min_n:
            stats[s] = (Xtr[m].mean(0), Xtr[m].std(0) + 1e-6)
    for s in np.unique(s_test):
        if s not in stats:
            m = s_test == s
            if m.sum() >= min_n:
                stats[s] = (Xte[m].mean(0), Xte[m].std(0) + 1e-6)
    for X, sites in ((Xtr, s_train), (Xte, s_test)):
        for s in np.unique(sites):
            m = sites == s
            mu, sd = stats.get(s, (g_mu, g_sd))
            X[m] = (X[m] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
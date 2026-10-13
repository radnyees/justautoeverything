"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization
(harmonization of site mean/scale shifts), L2 logistic regression."""
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
        return np.asarray(list(s), dtype=object)
    except Exception:
        pass
    try:
        return np.asarray([m["site"] for m in meta], dtype=object)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, s_train, F_test, s_test, min_n=5):
    Xtr = np.array(F_train, dtype=np.float64, copy=True)
    Xte = np.array(F_test, dtype=np.float64, copy=True)
    stats = {}
    for s in np.unique(s_train):
        m = s_train == s
        if m.sum() >= min_n:
            mu = Xtr[m].mean(0)
            sd = Xtr[m].std(0) + 1e-6
            stats[s] = (mu, sd)
    # global fallback
    g_mu, g_sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    for s in np.unique(s_train):
        m = s_train == s
        mu, sd = stats.get(s, (g_mu, g_sd))
        Xtr[m] = (Xtr[m] - mu) / sd
    for s in np.unique(s_test):
        m = s_test == s
        if s in stats:
            mu, sd = stats[s]
        elif m.sum() >= min_n:
            # unseen site: label-free standardization with its own statistics
            mu, sd = Xte[m].mean(0), Xte[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        Xte[m] = (Xte[m] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    if len(s_train) != len(F_train) or len(s_test) != len(F_test):
        s_train = np.zeros(len(F_train), dtype=object)
        s_test = np.zeros(len(F_test), dtype=object)
    Xtr, Xte = _site_standardize(F_train, s_train, F_test, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
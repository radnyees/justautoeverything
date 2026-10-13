"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site z-scoring (harmonization), L2 logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
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
        s = meta["site"]
        s = np.asarray(list(s), dtype=object)
    except Exception:
        try:
            s = np.asarray([m["site"] for m in meta], dtype=object)
        except Exception:
            s = np.zeros(n, dtype=object)
    if len(s) != n:
        s = np.zeros(n, dtype=object)
    return s.astype(str)


def _site_standardize(F_train, s_train, F_test, s_test, min_n=5):
    """Z-score each feature within site. Uses each set's own site stats (label-free);
    falls back to training-site stats, then global stats, for tiny groups."""
    g_mu = F_train.mean(0)
    g_sd = F_train.std(0) + 1e-6
    tr_stats = {}
    Xtr = np.empty_like(F_train)
    for s in np.unique(s_train):
        m = s_train == s
        if m.sum() >= min_n:
            mu, sd = F_train[m].mean(0), F_train[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        tr_stats[s] = (mu, sd)
        Xtr[m] = (F_train[m] - mu) / sd
    Xte = np.empty_like(F_test)
    for s in np.unique(s_test):
        m = s_test == s
        if m.sum() >= min_n:
            mu, sd = F_test[m].mean(0), F_test[m].std(0) + 1e-6
        elif s in tr_stats:
            mu, sd = tr_stats[s]
        else:
            mu, sd = g_mu, g_sd
        Xte[m] = (F_test[m] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, s_tr, F_test, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
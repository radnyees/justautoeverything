"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
(harmonization using training-fold site statistics), L2 logistic regression."""
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
        return np.zeros(n, dtype=object).astype(str)
    try:
        if hasattr(meta, "columns") and "site" in meta.columns:
            return meta["site"].astype(str).values
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
        return np.asarray([m["site"] for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object).astype(str)


def _site_standardize(F_train, F_test, s_train, s_test, eps=1e-6):
    Xtr = np.asarray(F_train, dtype=np.float64).copy()
    Xte = np.asarray(F_test, dtype=np.float64).copy()
    stats = {}
    for s in np.unique(s_train):
        m = s_train == s
        mu = Xtr[m].mean(axis=0)
        sd = Xtr[m].std(axis=0) if m.sum() > 2 else np.ones(Xtr.shape[1])
        sd = np.where(sd < eps, 1.0, sd)
        stats[s] = (mu, sd)
        Xtr[m] = (Xtr[m] - mu) / sd
    for s in np.unique(s_test):
        m = s_test == s
        if s in stats:
            mu, sd = stats[s]
        else:  # unseen site: label-free standardization with its own statistics
            mu = Xte[m].mean(axis=0)
            sd = Xte[m].std(axis=0) if m.sum() > 2 else np.ones(Xte.shape[1])
            sd = np.where(sd < eps, 1.0, sd)
        Xte[m] = (Xte[m] - mu) / sd
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
"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
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
        s = meta["site"].values if "site" in meta.columns else meta.iloc[:, 0].values
    elif isinstance(meta, pd.Series):
        s = meta.values
    elif isinstance(meta, dict):
        s = np.asarray(meta["site"])
    else:
        s = np.asarray([m["site"] if isinstance(m, dict) else m for m in meta])
    s = np.asarray(s).astype(str)
    return s if len(s) == n else np.zeros(n, dtype=object)


def _site_standardize(F_train, s_train, F_test, s_test, min_n=5):
    Ftr = np.array(F_train, dtype=np.float64)
    Fte = np.array(F_test, dtype=np.float64)
    gmu, gsd = Ftr.mean(0), Ftr.std(0) + 1e-6
    stats = {}
    for s in np.unique(s_train):
        m = s_train == s
        if m.sum() >= min_n:
            stats[s] = (Ftr[m].mean(0), Ftr[m].std(0) + 1e-6)
    for F, sites in ((Ftr, s_train), (Fte, s_test)):
        out = F.copy()
        for s in np.unique(sites):
            m = sites == s
            if s in stats:
                mu, sd = stats[s]
            elif m.sum() >= min_n:
                mu, sd = F[m].mean(0), F[m].std(0) + 1e-6
            else:
                mu, sd = gmu, gsd
            out[m] = (F[m] - mu) / sd
        F[:] = out
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, s_tr, F_test, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
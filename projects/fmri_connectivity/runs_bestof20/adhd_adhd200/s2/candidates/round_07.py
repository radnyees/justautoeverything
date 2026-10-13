"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization, L2 logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression


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
    except Exception:
        pass
    try:
        return np.asarray([m["site"] if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    Xtr = np.empty_like(F_train, dtype=np.float64)
    Xte = np.empty_like(F_test, dtype=np.float64)
    g_mu = F_train.mean(0)
    g_sd = F_train.std(0) + 1e-6
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        if mtr.sum() >= min_n:
            mu, sd = F_train[mtr].mean(0), F_train[mtr].std(0) + 1e-6
        elif mte.sum() >= min_n:
            mu, sd = F_test[mte].mean(0), F_test[mte].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        if mtr.any():
            Xtr[mtr] = (F_train[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (F_test[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
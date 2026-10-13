"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
(label-free ComBat-style location/scale harmonization), L2 logistic regression."""
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
            return np.asarray(meta["site"].astype(str).values)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray([str(s) for s in meta["site"]])
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test):
    """Z-score each feature within each site using all (label-free) items from that site."""
    Ftr = np.array(F_train, dtype=np.float64, copy=True)
    Fte = np.array(F_test, dtype=np.float64, copy=True)
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        X = np.vstack([Ftr[mtr], Fte[mte]])
        if X.shape[0] < 3:
            continue
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Ftr[mtr] = (Ftr[mtr] - mu) / sd
        Fte[mte] = (Fte[mte] - mu) / sd
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    if len(s_train) != len(F_train) or len(s_test) != len(F_test):
        s_train = np.zeros(len(F_train), dtype=object)
        s_test = np.zeros(len(F_test), dtype=object)
    Ftr, Fte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Ftr), y_train)
    return clf.predict_proba(sc.transform(Fte))[:, 1]
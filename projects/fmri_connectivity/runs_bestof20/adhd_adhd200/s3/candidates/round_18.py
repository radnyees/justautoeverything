"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
(label-free harmonization), L2 logistic regression."""
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
        if hasattr(meta, "columns") and "site" in meta.columns:
            return np.asarray(meta["site"].values).astype(str)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
        if isinstance(meta, (list, tuple)) and len(meta) == n:
            if isinstance(meta[0], dict):
                return np.asarray([m.get("site", "") for m in meta]).astype(str)
            return np.asarray(meta).astype(str)
        arr = np.asarray(meta)
        if arr.shape[0] == n:
            return arr.reshape(n, -1)[:, 0].astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test):
    Xtr = np.array(F_train, dtype=np.float64, copy=True)
    Xte = np.array(F_test, dtype=np.float64, copy=True)
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        pool = np.vstack([Xtr[mtr], Xte[mte]])
        if pool.shape[0] < 3:
            continue
        mu = pool.mean(0)
        sd = pool.std(0)
        sd[sd < 1e-6] = 1.0
        if mtr.any():
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_train = _sites(ctx.get("meta_train"), F_train.shape[0])
    s_test = _sites(ctx.get("meta_test"), F_test.shape[0])
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
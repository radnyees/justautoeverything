"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
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


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    Xtr = np.asarray(F_train, dtype=np.float64).copy()
    Xte = np.asarray(F_test, dtype=np.float64).copy()
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        if mtr.sum() >= min_n:
            ref = Xtr[mtr]
        elif mte.sum() >= min_n:
            ref = Xte[mte]  # label-free transductive fallback for unseen/rare sites
        else:
            ref = Xtr
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        if mtr.any():
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    n_tr, n_te = len(F_train), len(F_test)
    s_tr = _sites(ctx.get("meta_train"), n_tr)
    s_te = _sites(ctx.get("meta_test"), n_te)
    if len(s_tr) != n_tr or len(s_te) != n_te:
        s_tr = np.zeros(n_tr, dtype=object)
        s_te = np.zeros(n_te, dtype=object)
    Xtr, Xte = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
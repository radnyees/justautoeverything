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


def _site_standardize(X_tr, X_te, s_tr, s_te):
    X_tr = np.asarray(X_tr, dtype=np.float64).copy()
    X_te = np.asarray(X_te, dtype=np.float64).copy()
    out_tr = X_tr.copy()
    out_te = X_te.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr = s_tr == s
        mte = s_te == s
        ref = X_tr[mtr] if mtr.sum() >= 5 else (X_te[mte] if mte.sum() >= 2 else X_tr)
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        if mtr.any():
            out_tr[mtr] = (X_tr[mtr] - mu) / sd
        if mte.any():
            out_te[mte] = (X_te[mte] - mu) / sd
    return out_tr, out_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    if len(s_tr) != len(F_train) or len(s_te) != len(F_test):
        s_tr = np.zeros(len(F_train), dtype=object)
        s_te = np.zeros(len(F_test), dtype=object)
    X_tr, X_te = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization, L2 logistic regression."""
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
            return np.asarray(meta["site"].values).astype(str)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
        return np.asarray([m["site"] for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(X_tr, X_te, s_tr, s_te):
    """Z-score each feature within site using that site's training statistics."""
    X_tr = np.asarray(X_tr, dtype=np.float64).copy()
    X_te = np.asarray(X_te, dtype=np.float64).copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr = s_tr == s
        mte = s_te == s
        ref = X_tr[mtr] if mtr.sum() >= 5 else X_te[mte]  # label-free fallback for unseen sites
        if len(ref) < 2:
            ref = X_tr
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        if mtr.any():
            X_tr[mtr] = (X_tr[mtr] - mu) / sd
        if mte.any():
            X_te[mte] = (X_te[mte] - mu) / sd
    return X_tr, X_te


def fit_predict(F_train, y_train, F_test, ctx):
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    X_tr, X_te = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]
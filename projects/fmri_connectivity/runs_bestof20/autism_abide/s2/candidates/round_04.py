"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
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
            return np.asarray(meta["site"].values, dtype=object)
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(list(meta["site"]), dtype=object)
        return np.asarray([m["site"] if isinstance(m, dict) else m for m in meta], dtype=object)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    """Per-site z-scoring of each feature (label-free). Test items use their site's
    training statistics; sites unseen (or tiny) in training use their own statistics."""
    Ftr = np.array(F_train, dtype=np.float64, copy=True)
    Fte = np.array(F_test, dtype=np.float64, copy=True)
    for s in np.unique(np.concatenate([s_train, s_test]).astype(str)):
        mtr = s_train.astype(str) == s
        mte = s_test.astype(str) == s
        if mtr.sum() >= min_n:
            ref = F_train[mtr]
        elif mtr.sum() + mte.sum() >= 2:
            ref = np.concatenate([F_train[mtr], F_test[mte]], axis=0)
        else:
            continue
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0) + 1e-6
        if mtr.any():
            Ftr[mtr] = (F_train[mtr] - mu) / sd
        if mte.any():
            Fte[mte] = (F_test[mte] - mu) / sd
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    F_train, F_test = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
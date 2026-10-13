"""Pearson connectivity on CC200 (no GSR), Fisher z, upper triangle, per-site z-scoring, L2 logistic regression."""
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
        if "site" in meta.columns:
            return meta["site"].astype(str).to_numpy()
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]])
    try:
        return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    """Z-score features within site. Statistics from training items of the site
    (label-free); fall back to the test items' own statistics for unseen/tiny sites."""
    Ftr = np.array(F_train, dtype=np.float64, copy=True)
    Fte = np.array(F_test, dtype=np.float64, copy=True)
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        if mtr.sum() >= min_n:
            ref = Ftr[mtr]
        elif mtr.sum() + mte.sum() >= 2:
            ref = np.vstack([Ftr[mtr], Fte[mte]])
        else:
            continue
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        if mtr.any():
            Ftr[mtr] = (Ftr[mtr] - mu) / sd
        if mte.any():
            Fte[mte] = (Fte[mte] - mu) / sd
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Ftr, Fte = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Ftr), y_train)
    return clf.predict_proba(sc.transform(Fte))[:, 1]
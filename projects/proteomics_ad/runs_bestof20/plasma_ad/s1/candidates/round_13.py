"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    """Extract a batch label per item; fall back to a single batch if unavailable."""
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" in meta.columns:
                return meta["batch"].astype(str).values
            return np.zeros(n, dtype=object)
        if isinstance(meta, dict) and "batch" in meta:
            return np.asarray(meta["batch"]).astype(str)
        arr = np.asarray(meta)
        if arr.ndim == 1 and len(arr) == n:
            return arr.astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _batch_center(F_train, F_test, b_tr, b_te):
    """Subtract per-batch protein medians (NaN-aware), estimated from training items of that batch."""
    Ftr = F_train.copy()
    Fte = F_test.copy()
    for b in np.unique(b_tr):
        m = b_tr == b
        if m.sum() < 3:
            continue
        med = Ftr.loc[m].median(axis=0)
        glob = Ftr.median(axis=0)
        med = med.fillna(glob)
        Ftr.loc[m] = Ftr.loc[m] - med
        mt = b_te == b
        if mt.any():
            Fte.loc[mt] = Fte.loc[mt] - med
    for b in np.unique(b_te):                              # unseen batch: label-free self-centring
        if b in set(np.unique(b_tr)) and (b_tr == b).sum() >= 3:
            continue
        mt = b_te == b
        med = Fte.loc[mt].median(axis=0).fillna(0.0)
        Fte.loc[mt] = Fte.loc[mt] - med
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = F_train[keep], F_test[keep]
    b_tr = _batches(ctx.get("meta_train"), len(Ftr))
    b_te = _batches(ctx.get("meta_test"), len(Fte))
    if len(b_tr) == len(Ftr) and len(b_te) == len(Fte):
        Ftr, Fte = _batch_center(Ftr, Fte, b_tr, b_te)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
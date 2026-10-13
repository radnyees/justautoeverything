"""log2 + per-sample median centring + per-batch per-protein median centring + half-minimum imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batch_vec(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" in meta.columns:
                return meta["batch"].astype(str).values
            return meta.iloc[:, 0].astype(str).values
        if isinstance(meta, pd.Series):
            return meta.astype(str).values
        arr = np.asarray(meta)
        if arr.dtype.names and "batch" in arr.dtype.names:
            return arr["batch"].astype(str)
        if arr.ndim == 1 and len(arr) == n:
            if isinstance(arr[0], dict):
                return np.array([str(d.get("batch", "")) for d in arr])
            return arr.astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _batch_centre(Ftr, Fte, btr, bte):
    """Subtract per-protein median of each batch (observed values, train+test pooled, label-free)."""
    allF = pd.concat([Ftr, Fte], axis=0)
    allb = np.concatenate([btr, bte])
    grand = allF.median()
    Ctr, Cte = Ftr.copy(), Fte.copy()
    for b in np.unique(allb):
        sel = allb == b
        if sel.sum() < 5:
            continue
        shift = (allF[sel].median() - grand).fillna(0.0)
        mtr, mte = btr == b, bte == b
        if mtr.any():
            Ctr.loc[mtr] = Ctr.loc[mtr] - shift
        if mte.any():
            Cte.loc[mte] = Cte.loc[mte] - shift
    return Ctr, Cte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    btr = _batch_vec(ctx.get("meta_train"), len(F_train))
    bte = _batch_vec(ctx.get("meta_test"), len(F_test))
    Ftr, Fte = _batch_centre(F_train[keep], F_test[keep], btr, bte)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
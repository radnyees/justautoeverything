"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        if "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return np.zeros(n, dtype=object)
    try:
        return np.array([str(m.get("batch", 0)) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_centre(Ftr, Fte, btr, bte):
    """Per-protein median centring within each batch, using train+test of that batch (label-free)."""
    A = pd.concat([Ftr, Fte], axis=0, ignore_index=True)
    b = np.concatenate([btr, bte])
    grand = A.median(axis=0)
    out = A.copy()
    for g in np.unique(b):
        m = b == g
        if m.sum() < 5:
            continue
        med = A.loc[m].median(axis=0)
        shift = (med - grand).fillna(0.0)
        out.loc[m] = A.loc[m] - shift
    ntr = len(Ftr)
    Otr = out.iloc[:ntr]
    Ote = out.iloc[ntr:]
    Otr.index = Ftr.index
    Ote.index = Fte.index
    return Otr, Ote


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    btr = _batches(ctx.get("meta_train"), len(F_train))
    bte = _batches(ctx.get("meta_test"), len(F_test))
    Ftr, Fte = _batch_centre(F_train[keep], F_test[keep], btr, bte)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
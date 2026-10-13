"""log2 + per-sample median centring + per-batch per-protein median centring (label-free
harmonization) + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _get_batch(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns or len(meta) != len(items):
                return None
            if meta.index.equals(items.index):
                b = meta["batch"]
            elif set(items.index).issubset(set(meta.index)):
                b = meta.loc[items.index, "batch"]
            else:
                b = meta["batch"]
            return pd.Series(np.asarray(b), index=items.index)
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _get_batch(items, ctx)
    if batch is not None and batch.nunique() > 1:
        out = L.copy()
        grand = L.median(axis=0)                            # keep a common overall level
        for b in pd.unique(batch.values):
            idx = batch.index[batch.values == b]
            if len(idx) < 5:
                continue
            med = L.loc[idx].median(axis=0)                 # nan-aware per-protein batch median
            shift = (med - grand).fillna(0.0)
            out.loc[idx] = L.loc[idx].sub(shift, axis=1)
        L = out
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
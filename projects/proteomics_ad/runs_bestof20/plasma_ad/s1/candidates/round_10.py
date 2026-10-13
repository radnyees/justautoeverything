"""log2 + per-sample median centring + per-batch protein median centring + half-minimum imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_labels(items, ctx):
    meta = ctx.get("meta", None) if isinstance(ctx, dict) else None
    if meta is None or not isinstance(meta, pd.DataFrame) or "batch" not in meta.columns:
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        b = meta.loc[items.index, "batch"]
    elif len(meta) == len(items):
        b = meta["batch"]
    else:
        return None
    return pd.Series(np.asarray(b), index=items.index).astype(str)


def featurize(items, ctx):
    L = np.log2(items.astype(float).where(items > 0))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_labels(items, ctx)
    if batch is not None and batch.nunique() > 1:
        global_med = L.median(axis=0)
        out = L.copy()
        for b in batch.unique():
            idx = batch.index[batch == b]
            sub = L.loc[idx]
            med = sub.median(axis=0)
            cnt = sub.notna().sum(axis=0)
            med = med.where(cnt >= 3, global_med)          # too few values -> no batch shift
            shift = (med - global_med).fillna(0.0)
            out.loc[idx] = sub.sub(shift, axis=1)          # align batch medians to pooled median
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
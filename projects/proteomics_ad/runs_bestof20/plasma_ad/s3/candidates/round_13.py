"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not hasattr(meta, "columns") or "batch" not in meta.columns:
        return None
    b = meta["batch"]
    if len(b.index.intersection(items.index)) == len(items.index):
        return b.reindex(items.index).values
    if len(b) == len(items):
        return b.values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_vector(items, ctx)
    if batch is None:
        return L
    glob = L.median(axis=0)                                 # overall per-protein median (observed)
    out = L.copy()
    for bv in pd.unique(batch):
        m = batch == bv
        sub = L.loc[m]
        med = sub.median(axis=0)
        shift = (med - glob).fillna(0.0)                    # proteins unobserved in a batch: no shift
        out.loc[m] = sub.sub(shift, axis=1)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
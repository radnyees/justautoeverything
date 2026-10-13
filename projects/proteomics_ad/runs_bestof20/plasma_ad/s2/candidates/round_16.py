"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns:
                return None
            b = meta["batch"]
            if b.index.equals(items.index):
                return b.astype(str).values
            if len(b) == len(items):
                return b.astype(str).values
            b = b.reindex(items.index)
            return None if b.isna().any() else b.astype(str).values
        b = np.asarray(meta["batch"]).astype(str)
        return b if len(b) == len(items) else None
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_vector(items, ctx)
    if batch is not None and len(np.unique(batch)) > 1:
        out = L.copy()
        grand = L.median(axis=0)                           # keep overall protein level
        for b in np.unique(batch):
            m = batch == b
            if m.sum() < 5:
                continue
            bmed = L[m].median(axis=0)
            shift = (bmed - grand).fillna(0.0)             # per-protein batch offset (label-free)
            out.loc[m] = L.loc[m].sub(shift, axis=1)
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
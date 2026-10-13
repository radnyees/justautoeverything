"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not hasattr(meta, "columns") or "batch" not in meta.columns:
        return None
    try:
        b = meta.loc[items.index, "batch"]
    except Exception:
        if len(meta) != len(items):
            return None
        b = meta["batch"]
    return pd.Series(np.asarray(b), index=items.index).astype(str)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_labels(items, ctx)
    if b is not None and b.nunique() > 1:
        overall = L.median(axis=0)                         # keep global protein level
        for lev in b.unique():
            m = (b == lev).values
            if m.sum() < 5:
                continue
            bm = L.loc[m].median(axis=0)                   # NaN-aware, observed values only
            shift = (bm - overall).fillna(0.0)
            L.loc[m] = L.loc[m].sub(shift, axis=1)
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
"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None or not hasattr(meta, "columns") or "batch" not in meta.columns:
        return None
    if meta.index.equals(items.index) or set(items.index) <= set(meta.index):
        b = meta["batch"].reindex(items.index)
    elif len(meta) == len(items):
        b = pd.Series(meta["batch"].values, index=items.index)
    else:
        return None
    if b.isna().any():
        return None
    return b


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_labels(items, ctx)
    if b is not None and b.nunique() > 1:
        out = L.copy()
        for lev in b.unique():
            idx = b.index[b == lev]
            med = L.loc[idx].median(axis=0)                # per-protein batch median (observed values)
            out.loc[idx] = L.loc[idx].sub(med.fillna(0.0), axis=1)
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
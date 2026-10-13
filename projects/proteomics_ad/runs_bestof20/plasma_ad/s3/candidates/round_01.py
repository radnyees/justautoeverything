"""log2 + per-sample median centring + per-batch per-protein median centring + half-minimum imputation + L2 logistic regression."""
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
    return pd.Series(np.asarray(b), index=items.index)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_labels(items, ctx)
    if b is not None and b.nunique() > 1:
        # per-batch, per-protein median centring (label-free harmonization);
        # re-add the global protein median so the half-minimum fill stays meaningful
        glob = L.median(axis=0)
        out = L.copy()
        for lev in b.unique():
            m = (b == lev).values
            sub = L.loc[m]
            med = sub.median(axis=0)
            med = med.where(sub.notna().sum(axis=0) >= 3, glob)  # too few observations: no shift
            out.loc[m] = sub.sub(med, axis=1).add(glob, axis=1)
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
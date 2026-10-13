"""log2 + per-sample median centring + per-batch per-protein median centring + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame):
        if "batch" not in meta.columns:
            return None
        b = meta["batch"]
        if b.index.equals(items.index):
            return b.values
        try:
            return b.reindex(items.index).values if set(items.index) <= set(b.index) else (
                b.values if len(b) == len(items) else None)
        except Exception:
            return b.values if len(b) == len(items) else None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                     # per-sample median centring
    batch = _batch_vector(items, ctx)
    if batch is not None and pd.Series(batch).nunique(dropna=False) > 1:
        bs = pd.Series(batch, index=L.index).astype(str)
        out = L.copy()
        for g, idx in bs.groupby(bs).groups.items():
            sub = L.loc[idx]
            med = sub.median(axis=0)                        # per-protein, observed values only
            med = med.fillna(0.0)
            out.loc[idx] = sub - med
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
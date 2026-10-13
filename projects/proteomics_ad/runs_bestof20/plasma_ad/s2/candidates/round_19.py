"""log2 + per-sample median centring + per-batch per-protein median centring
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not isinstance(meta, pd.DataFrame) or "batch" not in meta.columns:
        return None
    b = meta["batch"]
    if b.index.equals(items.index):
        return b.values
    if items.index.isin(b.index).all():
        return b.reindex(items.index).values
    if len(b) == len(items):
        return b.values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_labels(items, ctx)
    if batch is not None:
        batch = pd.Series(batch, index=L.index).astype(str)
        grand = L.median(axis=0)                           # keep the global protein level
        for b in batch.unique():
            idx = batch.index[batch == b]
            if len(idx) < 5:
                continue
            L.loc[idx] = L.loc[idx] - L.loc[idx].median(axis=0) + grand
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
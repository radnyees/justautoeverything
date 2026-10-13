"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
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
            if "batch" not in meta.columns:
                return None
            b = meta["batch"]
            if b.index.equals(items.index):
                return b.values
            if set(items.index).issubset(set(b.index)):
                return b.reindex(items.index).values
            if len(b) == len(items):
                return b.values
        elif len(meta) == len(items):
            return np.asarray([m["batch"] if isinstance(m, dict) else m for m in meta])
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _get_batch(items, ctx)
    if batch is not None:
        batch = pd.Series(batch, index=L.index).astype(str)
        for b in batch.unique():
            m = (batch == b).values
            if m.sum() >= 5:
                L.loc[m] = L.loc[m] - L.loc[m].median(axis=0)  # per-batch protein centring (label-free)
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
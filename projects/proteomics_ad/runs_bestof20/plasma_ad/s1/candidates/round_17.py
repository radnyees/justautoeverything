"""log2 + per-sample median centring + per-batch protein median centring + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not isinstance(meta, pd.DataFrame) or "batch" not in meta.columns:
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        b = meta.loc[items.index, "batch"]
    elif len(meta) == len(items):
        b = pd.Series(meta["batch"].values, index=items.index)
    else:
        return None
    return b.astype(str).values


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None and len(np.unique(b)) > 1:
        overall = L.median(axis=0)
        for lev in np.unique(b):
            m = b == lev
            med = L.loc[m].median(axis=0)
            shift = (med - overall).fillna(0.0)            # per-protein batch offset (label-free)
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
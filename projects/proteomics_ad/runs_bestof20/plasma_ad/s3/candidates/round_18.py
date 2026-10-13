"""log2 + per-sample median centring + per-batch per-protein median centring
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not isinstance(meta, pd.DataFrame) or "batch" not in meta.columns:
        return None
    if meta.index.equals(items.index):
        return meta["batch"].values
    if set(items.index).issubset(set(meta.index)):
        return meta.loc[items.index, "batch"].values
    if len(meta) == len(items):
        return meta["batch"].values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None:
        b = pd.Series(b, index=L.index).astype(str)
        grand = L.median(axis=0)                            # keep overall protein level
        out = L.copy()
        for lev in b.unique():
            m = (b == lev).values
            if m.sum() < 3:
                continue
            med = L[m].median(axis=0)
            shift = (med - grand).fillna(0.0)
            out.loc[m] = L[m] - shift                       # label-free per-batch centring
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
"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None:
        return None
    meta = pd.DataFrame(meta)
    col = None
    for c in meta.columns:
        if str(c).lower() in ("batch", "site", "study"):
            col = c
            break
    if col is None:
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        try:
            return meta.loc[items.index, col].values
        except Exception:
            pass
    if len(meta) == len(items):
        return meta[col].values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None and len(pd.unique(b)) > 1:
        grand = L.median(axis=0)                           # keep overall protein level
        out = L.copy()
        for lev in pd.unique(b):
            m = (b == lev)
            med = L[m].median(axis=0)                      # observed-value median within batch
            shift = (med - grand).fillna(0.0)
            out.loc[m] = L[m] - shift
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
"""log2 + per-sample median centring + per-batch per-protein median centring
+ half-minimum imputation + L2 logistic regression."""
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
            col = "batch" if "batch" in meta.columns else ("site" if "site" in meta.columns else None)
            if col is None:
                return None
            if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
                b = meta.loc[items.index, col]
            elif len(meta) == len(items):
                b = meta[col]
            else:
                return None
            return np.asarray(b.astype(str).values)
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None and len(np.unique(b)) > 1:
        L = L.copy()
        for lev in np.unique(b):
            m = b == lev
            if m.sum() < 3:
                continue
            med = L.loc[m].median(axis=0)                  # per-batch protein median (NaN-skipping)
            L.loc[m] = L.loc[m].sub(med.fillna(0.0), axis=1)
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
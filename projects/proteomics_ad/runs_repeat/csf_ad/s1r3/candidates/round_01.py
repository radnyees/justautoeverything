"""log2 + per-sample median centring + per-site protein median centring + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    meta = pd.DataFrame(meta)
    col = None
    for c in ("site", "batch", "cohort", "study"):
        if c in meta.columns:
            col = c
            break
    if col is None:
        return None
    try:
        s = meta[col].reindex(items.index)
        if s.isna().all():
            raise KeyError
    except Exception:
        if len(meta) != len(items):
            return None
        s = pd.Series(meta[col].values, index=items.index)
    return s.fillna("NA").astype(str)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_labels(items, ctx)
    if site is not None:
        # label-free harmonisation: remove per-site protein offsets (observed values only)
        site_med = L.groupby(site.values).transform("median")
        L = L - site_med.fillna(0.0)
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
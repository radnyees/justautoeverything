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
    if "site" not in meta.columns or len(meta) != len(items):
        return None
    if meta.index.equals(items.index):
        return meta["site"].astype(str).values
    try:
        s = meta["site"].reindex(items.index)
        if s.notna().all():
            return s.astype(str).values
    except Exception:
        pass
    return meta["site"].astype(str).values


def featurize(items, ctx):
    L = np.log2(items.astype(float).where(items.astype(float) > 0))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _site_labels(items, ctx)
    if sites is not None:
        L = L.copy()
        for s in np.unique(sites):
            m = sites == s
            med = L.loc[m].median(axis=0)                   # per-site protein median (label-free)
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
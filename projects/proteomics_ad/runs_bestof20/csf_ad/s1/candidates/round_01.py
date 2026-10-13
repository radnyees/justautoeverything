"""log2 + per-sample median centring + per-site per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or not isinstance(meta, pd.DataFrame) or "site" not in meta.columns:
        return None
    try:
        if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
            return meta.loc[items.index, "site"].astype(str).values
    except Exception:
        pass
    if len(meta) == len(items):
        return meta["site"].astype(str).values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _site_labels(items, ctx)
    if sites is not None:
        out = L.copy()
        for s in np.unique(sites):
            m = sites == s
            med = L.loc[m].median(axis=0)                   # per-site protein median (observed values)
            out.loc[m] = L.loc[m] - med.fillna(0.0)
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
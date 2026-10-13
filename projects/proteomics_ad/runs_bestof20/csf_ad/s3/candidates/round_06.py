"""log2 + per-sample median centring + per-site per-protein median centring (harmonization)
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta", None) if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "site" not in meta.columns:
                return None
            if meta.index.equals(items.index):
                return meta["site"].astype(str).values
            if len(meta) == len(items):
                return meta["site"].astype(str).values
            m = meta.reindex(items.index)
            if m["site"].notna().all():
                return m["site"].astype(str).values
            return None
        s = pd.DataFrame(meta)
        if "site" in s.columns and len(s) == len(items):
            return s["site"].astype(str).values
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _site_labels(items, ctx)
    if sites is not None:
        L = L.copy()
        for s in np.unique(sites):
            idx = sites == s
            if idx.sum() < 3:
                continue
            med = L.loc[idx].median(axis=0)                 # per-site, per-protein median (observed)
            L.loc[idx] = L.loc[idx].sub(med.fillna(0.0), axis=1)
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
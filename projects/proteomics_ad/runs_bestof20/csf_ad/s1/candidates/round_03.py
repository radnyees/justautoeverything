"""log2 + per-sample median centring + per-site protein median centring + half-minimum imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "site" not in getattr(meta, "columns", []):
        return None
    site = meta["site"]
    if site.index.equals(items.index):
        return site.values
    try:
        return site.reindex(items.index).values
    except Exception:
        return site.values if len(site) == len(items) else None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_vector(items, ctx)
    if site is not None:
        site = pd.Series(site, index=L.index).astype(str)
        for s in site.unique():
            m = (site == s).values
            med = L.loc[m].median(axis=0)                  # label-free per-site protein location
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
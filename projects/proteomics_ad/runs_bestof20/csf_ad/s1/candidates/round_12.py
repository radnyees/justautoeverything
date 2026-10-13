"""log2 + per-sample median centring + per-site half-min imputation + per-site protein centring + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None or "site" not in getattr(meta, "columns", []):
        return None
    if isinstance(meta, pd.DataFrame) and meta.index.equals(items.index):
        return meta["site"].astype(str).values
    if len(meta) == len(items):
        return np.asarray(meta["site"]).astype(str)
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    L = L.loc[:, L.notna().mean() >= 0.7]                  # label-free filter: quantified in >= 70%
    gfill = L.min() - 1.0
    site = _site_labels(items, ctx)
    if site is None:
        return L.fillna(gfill)
    out = L.copy()
    for s in np.unique(site):
        m = site == s
        block = L.loc[m]
        fill = (block.min() - 1.0).fillna(gfill)           # half-minimum within site
        block = block.fillna(fill)
        out.loc[m] = block - block.median()                # per-site protein centring
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = F_train.values
    Xte = F_test.values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
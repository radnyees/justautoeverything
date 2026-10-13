"""log2 + per-sample median centring + per-site protein median centring + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_series(ctx, index):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    meta = pd.DataFrame(meta)
    col = "site" if "site" in meta.columns else ("batch" if "batch" in meta.columns else None)
    if col is None:
        return None
    if len(meta) == len(index):
        if meta.index.equals(index):
            return meta[col]
        return pd.Series(meta[col].values, index=index)
    try:
        return meta[col].reindex(index)
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_series(ctx, L.index)
    if site is not None and site.notna().all():
        # label-free per-site harmonisation: centre each protein on its within-site median
        med = L.groupby(site.values).transform("median")
        L = L - med.fillna(0.0)
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # low-value imputation on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
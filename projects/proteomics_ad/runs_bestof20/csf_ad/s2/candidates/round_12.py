"""log2 + per-sample median centring + per-site protein standardisation + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or not isinstance(meta, pd.DataFrame) or "site" not in meta.columns:
        return None
    if items.index.isin(meta.index).all():
        return meta.loc[items.index, "site"].astype(str).values
    if len(meta) == len(items):
        return meta["site"].astype(str).values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float).where(items.astype(float) > 0))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_vector(items, ctx)
    if site is None:
        return L
    out = L.copy()
    for s in np.unique(site):
        m = site == s
        block = L.loc[m]
        med = block.median(axis=0)
        sd = block.std(axis=0)
        sd = sd.where((sd > 1e-6) & (block.notna().sum(axis=0) >= 3), np.nan)
        sd = sd.fillna(sd.median() if np.isfinite(sd.median()) else 1.0)
        out.loc[m] = (block - med) / sd                    # label-free per-site harmonisation
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # low-value (MNAR-style) imputation
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
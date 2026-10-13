"""log2 + per-sample median centring + per-site robust protein standardization
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            col = "site" if "site" in meta.columns else ("batch" if "batch" in meta.columns else None)
            if col is None:
                return None
            if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
                s = meta.loc[items.index, col]
            else:
                s = meta[col]
            s = np.asarray(s)
        else:
            s = np.asarray([m.get("site", m.get("batch")) for m in meta])
        if len(s) != len(items):
            return None
        return s
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float).where(items.astype(float) > 0))
    L = L.sub(L.median(axis=1), axis=0)                     # per-sample median centring
    site = _site_vector(items, ctx)
    if site is None:
        return L
    out = L.copy()
    for s in pd.unique(site):
        m = site == s
        block = L.loc[m]
        med = block.median()
        q75, q25 = block.quantile(0.75), block.quantile(0.25)
        scale = (q75 - q25) / 1.349
        sd = block.std()
        scale = scale.where(scale > 1e-6, sd)
        scale = scale.where(scale > 1e-6, 1.0).fillna(1.0)
        out.loc[m] = (block - med) / scale                 # per-site robust z-score
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]   # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                         # below-minimum imputation
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
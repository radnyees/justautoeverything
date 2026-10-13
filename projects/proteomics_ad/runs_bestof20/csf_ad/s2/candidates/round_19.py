"""log2 + per-sample median centring + per-site per-protein median centring (label-free harmonization)
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_series(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    meta = pd.DataFrame(meta)
    col = "site" if "site" in meta.columns else ("batch" if "batch" in meta.columns else None)
    if col is None:
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        s = meta.loc[items.index, col]
    elif len(meta) == len(items):
        s = pd.Series(meta[col].values, index=items.index)
    else:
        return None
    return s.astype(str)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_series(items, ctx)
    if site is not None:
        out = L.copy()
        for s in site.unique():
            idx = site.index[site.values == s]
            if len(idx) < 3:
                continue
            med = L.loc[idx].median(axis=0, skipna=True)   # per-site, per-protein median (label-free)
            out.loc[idx] = L.loc[idx].sub(med.fillna(0.0), axis=1)
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
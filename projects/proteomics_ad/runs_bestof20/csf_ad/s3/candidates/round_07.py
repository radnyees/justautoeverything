"""log2 + per-sample median centring + half-minimum imputation + per-site median centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            col = "site" if "site" in meta.columns else ("batch" if "batch" in meta.columns else None)
            if col is None:
                return None
            if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
                s = meta.loc[items.index, col]
            elif len(meta) == len(items):
                s = meta[col]
            else:
                return None
            return np.asarray(s.astype(str).values)
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                     # per-sample median centring
    keep = L.columns[L.notna().mean() >= 0.7]               # label-free filter on all items
    L = L[keep]
    L = L.fillna(L.min() - 1.0)                             # half-minimum on the log2 scale
    site = _site_vector(items, ctx)
    if site is not None:
        X = L.values.copy()
        for s in np.unique(site):
            m = site == s
            if m.sum() >= 3:
                X[m] -= np.median(X[m], axis=0)             # per-site median centring
        L = pd.DataFrame(X, index=L.index, columns=L.columns)
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = np.asarray(F_train, dtype=float)
    Xte = np.asarray(F_test, dtype=float)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
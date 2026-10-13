"""log2 + per-sample median centring + per-site per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            s = meta["site"]
            if s.index.equals(items.index):
                return s.astype(str).values
            if set(items.index).issubset(set(s.index)):
                return s.reindex(items.index).astype(str).values
            if len(s) == len(items):
                return s.astype(str).values
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _site_labels(items, ctx)
    if sites is None:
        return L
    out = L.copy()
    for s in np.unique(sites):
        m = sites == s
        if m.sum() < 5:
            continue
        med = L.loc[m].median(axis=0)                       # per-site, per-protein median (label-free)
        med = med.fillna(0.0)
        out.loc[m] = L.loc[m].sub(med, axis=1)
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
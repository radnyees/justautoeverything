"""log2 + per-sample median centring + per-site protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sites(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "site" not in meta.columns:
                return None
            s = meta["site"]
            if s.index.equals(items.index):
                return s.values
            if len(s) == len(items):
                return s.values
            return s.reindex(items.index).values
        s = np.asarray(meta["site"])
        return s if len(s) == len(items) else None
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _sites(items, ctx)
    if sites is not None:
        sites = pd.Series(sites, index=L.index).astype(str)
        gmed = L.median(axis=0)
        for s in sites.unique():
            m = (sites == s).values
            if m.sum() < 5:
                continue
            smed = L.loc[m].median(axis=0)
            shift = (smed - gmed).fillna(0.0)               # per-site protein offset (label-free)
            L.loc[m] = L.loc[m] - shift
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
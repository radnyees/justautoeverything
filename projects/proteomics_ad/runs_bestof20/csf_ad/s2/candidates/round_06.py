"""log2 + per-sample median centring + per-site per-protein median centring (harmonization)
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_series(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame):
        if "site" in meta.columns:
            s = meta["site"]
        elif "batch" in meta.columns:
            s = meta["batch"]
        else:
            return None
        if s.index.equals(items.index):
            return s
        try:
            r = s.reindex(items.index)
            if r.notna().all():
                return r
        except Exception:
            pass
        if len(s) == len(items):
            return pd.Series(np.asarray(s), index=items.index)
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_series(items, ctx)
    if site is not None:
        # per-site, per-protein median centring (label-free harmonization)
        site_med = L.groupby(site.values).transform("median")
        glob_med = L.median(axis=0)
        L = L - site_med + glob_med                        # keep global level for half-min fill
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
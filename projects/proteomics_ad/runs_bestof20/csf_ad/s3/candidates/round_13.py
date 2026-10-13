"""log2 + per-sample median centring + per-site protein median centring + half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame):
        if "site" not in meta.columns:
            return None
        s = meta["site"]
        if s.index.equals(items.index):
            return s.values
        try:
            if items.index.isin(s.index).all():
                return s.loc[items.index].values
        except Exception:
            pass
        if len(s) == len(items):
            return s.values
        return None
    try:
        s = np.asarray(meta["site"])
        return s if len(s) == len(items) else None
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_labels(items, ctx)
    if site is not None:                                   # label-free per-site harmonization
        site = pd.Series(site, index=L.index).astype(str)
        med = L.groupby(site).transform("median")          # per-site, per-protein median (NaN-ignoring)
        L = L - med.fillna(0.0)
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
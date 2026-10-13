"""log2 + per-sample median centring + per-site per-protein median centring (harmonization)
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "site" not in getattr(meta, "columns", []):
        return None
    site = meta["site"]
    if isinstance(site, pd.Series) and site.index.equals(items.index):
        return site.astype(str).values
    if len(site) == len(items):
        return np.asarray(site).astype(str)
    try:
        return site.reindex(items.index).astype(str).values
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    sites = _site_labels(items, ctx)
    if sites is not None:
        # label-free per-site harmonization: remove each cohort's per-protein median offset
        gmed = L.median(axis=0)
        out = L.copy()
        for s in np.unique(sites):
            m = sites == s
            smed = L.loc[m].median(axis=0)
            smed = smed.fillna(gmed)                       # proteins never seen in a site
            out.loc[m] = L.loc[m] - smed + gmed            # keep global level (for imputation floor)
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
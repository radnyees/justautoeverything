"""log2 + per-sample median centring + per-site per-protein median centring (harmonization)
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "site" not in meta.columns:
                return None
            if meta.index.equals(items.index):
                s = meta["site"]
            elif len(meta) == len(items):
                s = pd.Series(meta["site"].values, index=items.index)
            else:
                s = meta["site"].reindex(items.index)
        else:
            s = pd.Series(np.asarray(meta["site"]), index=items.index)
    except Exception:
        return None
    if s.isna().any():
        s = s.astype(object).where(s.notna(), "__unknown__")
    return s.astype(str)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    site = _site_labels(items, ctx)
    if site is not None and site.nunique() > 1:
        # remove cohort-level offsets per protein (label-free harmonization);
        # scoring is within site, so only within-site contrasts matter
        med = L.groupby(site.values).transform("median")
        L = L - med
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # shifted minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
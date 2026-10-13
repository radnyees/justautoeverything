"""log2 + per-sample median centring + half-minimum imputation + per-site feature centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _sites(ctx, key, n):
    meta = ctx.get(key)
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            return meta["site"].astype(str).values
        return np.asarray([m["site"] for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Remove per-site feature means (label-free; uses all samples of the site)."""
    X = np.vstack([Xtr, Xte])
    s = np.concatenate([s_tr, s_te])
    Xc = X.copy()
    for site in np.unique(s):
        m = s == site
        Xc[m] -= X[m].mean(axis=0)
    return Xc[:len(Xtr)], Xc[len(Xtr):]


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    s_tr = _sites(ctx, "meta_train", len(Xtr))
    s_te = _sites(ctx, "meta_test", len(Xte))
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
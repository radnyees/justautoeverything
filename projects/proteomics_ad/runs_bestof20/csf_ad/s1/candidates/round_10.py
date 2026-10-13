"""log2 + per-sample median centring + half-minimum imputation + per-site feature centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        col = "site" if "site" in meta.columns else ("batch" if "batch" in meta.columns else None)
        if col is None:
            return np.zeros(n, dtype=object)
        return meta[col].astype(str).values
    return np.asarray(meta).astype(str).reshape(n, -1)[:, 0]


def _site_center(Xtr, Xte, s_tr, s_te):
    """Subtract per-site feature medians (estimated on the site's training samples; label-free)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        if mtr.sum() >= 3:
            med = np.median(Xtr[mtr], axis=0)
        elif mte.sum() >= 3:
            med = np.median(Xte[mte], axis=0)               # unseen site: use its own (label-free) stats
        else:
            continue
        Xtr[mtr] -= med
        Xte[mte] -= med
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
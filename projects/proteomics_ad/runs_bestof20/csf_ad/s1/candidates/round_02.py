"""log2 + per-sample median centring + half-minimum imputation + per-site protein centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _site_vector(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        if "site" in meta.columns:
            return meta["site"].astype(str).values
        if "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return np.zeros(n, dtype=object)
    return np.asarray(meta).astype(str).reshape(-1)[:n]


def _site_center(Xtr, Xte, s_tr, s_te):
    """Subtract per-site protein medians (train-fold stats; test-own fallback for unseen sites)."""
    Xtr_c = Xtr.copy()
    Xte_c = Xte.copy()
    for s in np.unique(s_tr):
        m = np.median(Xtr[s_tr == s], axis=0)
        Xtr_c[s_tr == s] -= m
        if np.any(s_te == s):
            Xte_c[s_te == s] -= m
    for s in np.unique(s_te):
        if not np.any(s_tr == s):
            Xte_c[s_te == s] -= np.median(Xte[s_te == s], axis=0)
    return Xtr_c, Xte_c


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    s_tr = _site_vector(ctx.get("meta_train"), len(Xtr))
    s_te = _site_vector(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
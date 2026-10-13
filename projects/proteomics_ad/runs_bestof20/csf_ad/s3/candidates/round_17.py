"""log2 + per-sample median centring + half-minimum imputation + per-site median centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _site_labels(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        for col in ("site", "batch", "study"):
            if col in meta.columns:
                return meta[col].astype(str).values
        return np.zeros(n, dtype=object)
    return np.asarray(meta).astype(str).reshape(-1)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Subtract per-site feature medians (estimated on train; test-own medians for unseen sites)."""
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    meds = {}
    for s in np.unique(s_tr):
        m = s_tr == s
        meds[s] = np.median(Xtr[m], axis=0)
        Xtr[m] -= meds[s]
    for s in np.unique(s_te):
        m = s_te == s
        if s in meds:
            Xte[m] -= meds[s]
        else:
            Xte[m] -= np.median(Xte[m], axis=0)
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    s_tr = _site_labels(ctx.get("meta_train"), len(Xtr))
    s_te = _site_labels(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)          # per-site harmonization

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
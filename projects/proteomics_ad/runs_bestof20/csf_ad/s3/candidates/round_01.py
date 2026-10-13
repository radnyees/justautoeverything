"""log2 + per-sample median centring + half-minimum imputation + per-site centring + L2 logistic regression."""
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
        if "site" in meta.columns:
            return meta["site"].astype(str).values
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray(meta["site"]).astype(str)
    return np.zeros(n, dtype=object)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Subtract per-site feature means estimated on training samples (harmonization only)."""
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    gmean = Xtr.mean(axis=0)
    means = {}
    for s in np.unique(s_tr):
        m = s_tr == s
        means[s] = Xtr[m].mean(axis=0) if m.sum() >= 3 else gmean
        Xtr_c[m] -= means[s]
    for s in np.unique(s_te):
        m = s_te == s
        if s in means:
            Xte_c[m] -= means[s]
        elif m.sum() >= 3:                                   # unseen site: label-free own-site centring
            Xte_c[m] -= Xte[m].mean(axis=0)
        else:
            Xte_c[m] -= gmean
    return Xtr_c, Xte_c


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
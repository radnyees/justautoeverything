"""log2 + per-sample median centring + half-minimum imputation + per-site median centring + L2 logistic regression."""
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
    if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
        return meta["site"].astype(str).values
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray(meta["site"]).astype(str)
    return np.zeros(n, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    # label-free per-site median centring (harmonization only; scored within site)
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te])
    for s in np.unique(s_all):
        m = s_all == s
        X_all[m] -= np.median(X_all[m], axis=0)
    Xtr, Xte = X_all[:len(Xtr)], X_all[len(Xtr):]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
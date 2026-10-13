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
    if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
        return np.asarray(meta["site"]).astype(str)
    try:
        return np.asarray(pd.DataFrame(meta)["site"]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Remove per-site protein means (label-free harmonization; scoring is within site)."""
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr = s_tr == s
        mte = s_te == s
        if mtr.sum() >= 3:
            mu = Xtr[mtr].mean(axis=0)
        elif mte.sum() >= 3:
            mu = Xte[mte].mean(axis=0)
        else:
            continue
        Xtr[mtr] -= mu
        Xte[mte] -= mu
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
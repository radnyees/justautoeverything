"""log2 + per-sample median centring + half-minimum imputation + per-site protein centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _sites(meta, n):
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            s = np.asarray(meta["site"]).astype(str)
        else:
            s = np.asarray(meta).astype(str).ravel()
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.array(["_all"] * n)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Label-free harmonization: subtract per-site median of each protein (train+test pooled)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        pooled = np.vstack([Xtr[mtr], Xte[mte]])
        if pooled.shape[0] < 3:
            continue
        med = np.median(pooled, axis=0)
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
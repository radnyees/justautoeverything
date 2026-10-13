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
        s = np.asarray(pd.DataFrame(meta)["site"]).astype(str)
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _site_center(Xtr, Xte, s_tr, s_te):
    """Label-free harmonization: subtract each site's per-protein mean (train+test pooled)."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        block = np.vstack([Xtr[mtr], Xte[mte]])
        if block.shape[0] < 3:
            continue
        mu = block.mean(axis=0)
        Xtr[mtr] -= mu
        Xte[mte] -= mu
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    s_tr = _sites(ctx, "meta_train", Xtr.shape[0])
    s_te = _sites(ctx, "meta_test", Xte.shape[0])
    Xtr, Xte = _site_center(Xtr, Xte, s_tr, s_te)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
"""log2 + per-sample median centring + per-site protein median centring + half-min imputation + L2 logistic regression."""
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
    try:
        return np.asarray([m.get("site", 0) if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _site_center(F_train, F_test, s_tr, s_te):
    """Subtract per-site protein medians (estimated on training rows of that site; label-free)."""
    Ftr, Fte = F_train.copy(), F_test.copy()
    glob = F_train.median()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        if mtr.sum() >= 5:
            med = F_train[mtr].median()
        elif mte.sum() >= 5:
            med = F_test[mte].median()                      # unseen site: centre on its own unlabeled rows
        else:
            med = glob
        med = med.fillna(glob)
        shift = med - glob
        if mtr.any():
            Ftr.loc[mtr] = F_train[mtr] - shift
        if mte.any():
            Fte.loc[mte] = F_test[mte] - shift
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = _site_center(F_train[keep], F_test[keep], s_tr, s_te)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
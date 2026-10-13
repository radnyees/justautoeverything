"""log2 + per-sample median centring + per-site protein median centring + half-min imputation + L2 LR."""
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


def _site_centre(F_train, F_test, s_tr, s_te):
    """Subtract per-site, per-protein median (estimated on train; fallback to test's own)."""
    Ftr = F_train.copy()
    Fte = F_test.copy()
    glob = F_train.median()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr = s_tr == s
        mte = s_te == s
        if mtr.sum() >= 3:
            med = F_train[mtr].median()
        elif mte.sum() >= 3:
            med = F_test[mte].median()
        else:
            med = glob
        med = med.fillna(glob).fillna(0.0)
        if mtr.any():
            Ftr.loc[mtr] = F_train[mtr] - med
        if mte.any():
            Fte.loc[mte] = F_test[mte] - med
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    Ftr, Fte = _site_centre(F_train[keep], F_test[keep], s_tr, s_te)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
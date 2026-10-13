"""log2 + per-sample median centring + half-minimum imputation + per-site centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _sites(meta, n):
    try:
        if meta is not None and "site" in meta:
            s = np.asarray(meta["site"]).astype(str)
            if len(s) == n:
                return s
    except Exception:
        pass
    return np.array(["all"] * n)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)

    # per-site centring (harmonization only): remove cohort offsets estimated on training samples
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        if mtr.sum() >= 3:
            mu = Xtr[mtr].mean(axis=0)
        elif mte.sum() >= 3:
            mu = Xte[mte].mean(axis=0)                      # unseen site: label-free test-side centring
        else:
            mu = Xtr.mean(axis=0)
        Xtr_c[mtr] -= mu
        Xte_c[mte] -= mu

    sc = StandardScaler().fit(Xtr_c)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr_c), y_train)
    return clf.predict_proba(sc.transform(Xte_c))[:, 1]
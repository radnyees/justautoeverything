"""log2 + per-sample median centring + half-minimum imputation + per-batch protein centring + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        if "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "batch" in meta:
        return np.asarray(meta["batch"]).astype(str)
    try:
        return np.asarray([m.get("batch", 0) if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    # per-batch centring of each protein (harmonization only; stats from the training fold)
    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    gmed = np.median(Xtr, axis=0)
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    for b in np.unique(np.concatenate([btr, bte])):
        mtr, mte = btr == b, bte == b
        if mtr.sum() >= 5:
            off = np.median(Xtr[mtr], axis=0) - gmed
        elif mte.sum() >= 5:                                # unseen batch: label-free own-batch stats
            off = np.median(Xte[mte], axis=0) - gmed
        else:
            off = 0.0
        Xtr_c[mtr] -= off
        Xte_c[mte] -= off

    sc = StandardScaler().fit(Xtr_c)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr_c), y_train)
    return clf.predict_proba(sc.transform(Xte_c))[:, 1]
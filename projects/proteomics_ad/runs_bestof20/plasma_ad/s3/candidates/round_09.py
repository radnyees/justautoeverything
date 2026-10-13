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
    try:
        b = np.asarray(meta["batch"]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)
    return b if len(b) == n else np.zeros(n, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)

    # per-batch, per-protein median centring (label-free harmonization; train stats)
    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    for b in np.unique(np.concatenate([btr, bte])):
        mtr, mte = btr == b, bte == b
        if mtr.sum() >= 5:
            med = np.median(Xtr[mtr], axis=0)
        elif mte.sum() >= 5:
            med = np.median(Xte[mte], axis=0)               # unseen batch: use its own (label-free) centre
        else:
            continue
        Xtr[mtr] -= med
        Xte[mte] -= med

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
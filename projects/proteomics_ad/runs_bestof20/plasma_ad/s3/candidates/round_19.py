"""log2 + per-sample median centring + per-batch per-protein median centring (train-derived)
+ half-minimum imputation + L2 logistic regression."""
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
        b = np.asarray(pd.DataFrame(meta)["batch"]).astype(str)
        if len(b) == n:
            return b
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _batch_center(F_train, F_test, b_tr, b_te):
    """Subtract per-batch protein medians estimated on training samples of that batch."""
    Xtr = F_train.copy()
    Xte = F_test.copy()
    global_med = F_train.median()
    for b in np.unique(np.concatenate([b_tr, b_te])):
        mtr = b_tr == b
        mte = b_te == b
        if mtr.sum() >= 5:
            med = F_train[mtr].median()
            med = med.fillna(global_med)                   # protein unobserved in this batch
        else:
            med = global_med                               # unseen / tiny batch: no correction
        shift = (med - global_med).fillna(0.0)
        if mtr.any():
            Xtr.loc[mtr] = F_train[mtr] - shift
        if mte.any():
            Xte.loc[mte] = F_test[mte] - shift
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    b_tr = _batches(ctx.get("meta_train"), len(F_train))
    b_te = _batches(ctx.get("meta_test"), len(F_test))
    Ftr, Fte = _batch_center(F_train[keep], F_test[keep], b_tr, b_te)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
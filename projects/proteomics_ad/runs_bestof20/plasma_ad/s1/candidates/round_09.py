"""log2 + per-sample median centring + half-minimum imputation + per-batch protein median
centring (label-free, train-fold statistics) + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(ctx, key, n):
    meta = ctx.get(key)
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns:
                return None
            b = meta["batch"].astype(str).values
        else:
            b = np.asarray([m["batch"] for m in meta]).astype(str)
    except Exception:
        return None
    return b if len(b) == n else None


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    # per-batch, per-protein median centring using training-fold statistics only
    btr = _batches(ctx, "meta_train", Xtr.shape[0])
    bte = _batches(ctx, "meta_test", Xte.shape[0])
    if btr is not None and bte is not None:
        Xtr = Xtr.copy()
        Xte = Xte.copy()
        global_med = np.median(Xtr, axis=0)
        meds = {b: np.median(Xtr[btr == b], axis=0) for b in np.unique(btr)}
        for b, m in meds.items():
            Xtr[btr == b] -= m
        for b in np.unique(bte):
            Xte[bte == b] -= meds.get(b, global_med)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
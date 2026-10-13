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
    if isinstance(meta, pd.Series):
        return meta.astype(str).values
    try:
        return np.asarray([str(m.get("batch", 0)) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_center(Xtr, Xte, btr, bte):
    """Remove per-batch protein means (label-free; uses train+test features of each batch)."""
    X = np.vstack([Xtr, Xte])
    b = np.concatenate([btr, bte])
    grand = X.mean(axis=0)
    Xc = X.copy()
    for g in np.unique(b):
        m = b == g
        if m.sum() >= 3:
            Xc[m] = X[m] - X[m].mean(axis=0) + grand
    return Xc[: len(Xtr)], Xc[len(Xtr):]


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    if len(btr) == len(Xtr) and len(bte) == len(Xte):
        Xtr, Xte = _batch_center(Xtr, Xte, btr, bte)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
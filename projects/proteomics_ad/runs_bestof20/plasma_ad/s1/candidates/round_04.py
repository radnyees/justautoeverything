"""log2 + per-sample median centring + per-batch protein median centring + half-minimum imputation + L2 logistic regression."""
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
            return np.asarray(meta["batch"]).astype(str)
        return np.asarray(meta.iloc[:, 0]).astype(str)
    if isinstance(meta, pd.Series):
        return np.asarray(meta).astype(str)
    try:
        return np.asarray([m["batch"] if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_centre(F_train, F_test, ctx):
    """Subtract each protein's per-batch median (observed values, train+test pooled; label-free)."""
    btr = _batches(ctx.get("meta_train"), len(F_train))
    bte = _batches(ctx.get("meta_test"), len(F_test))
    if len(btr) != len(F_train) or len(bte) != len(F_test):
        return F_train, F_test
    A = pd.concat([F_train, F_test], axis=0, ignore_index=True)
    b = np.concatenate([btr, bte])
    grand = A.median(axis=0)
    out = A.copy()
    for g in np.unique(b):
        m = b == g
        med = A[m].median(axis=0).fillna(grand)
        out.loc[m] = A[m] - med + grand
    Ftr = out.iloc[:len(F_train)].set_axis(F_train.index)
    Fte = out.iloc[len(F_train):].set_axis(F_test.index)
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = _batch_centre(F_train[keep], F_test[keep], ctx)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
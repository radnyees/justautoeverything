"""log2 + per-sample median centring + per-batch protein median centring (train-derived)
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
    if isinstance(meta, pd.DataFrame):
        if "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return meta.iloc[:, 0].astype(str).values
    if isinstance(meta, pd.Series):
        return meta.astype(str).values
    try:
        return np.asarray([str(m.get("batch")) if isinstance(m, dict) else str(m) for m in meta])
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_center(F_train, F_test, b_tr, b_te):
    """Subtract per-batch protein medians estimated on the training samples of that batch."""
    Ftr = F_train.copy()
    Fte = F_test.copy()
    glob = F_train.median()
    for b in np.unique(np.concatenate([b_tr, b_te])):
        mtr = b_tr == b
        if mtr.sum() >= 5:
            med = F_train.loc[mtr].median()
            med = med.fillna(glob)
        else:
            med = glob
        shift = (med - glob).fillna(0.0)
        if mtr.any():
            Ftr.loc[mtr] = F_train.loc[mtr] - shift
        mte = b_te == b
        if mte.any():
            Fte.loc[mte] = F_test.loc[mte] - shift
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    b_tr = _batches(ctx.get("meta_train"), len(F_train))
    b_te = _batches(ctx.get("meta_test"), len(F_test))
    if len(b_tr) == len(F_train) and len(b_te) == len(F_test):
        F_train, F_test = _batch_center(F_train, F_test, np.asarray(b_tr), np.asarray(b_te))

    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
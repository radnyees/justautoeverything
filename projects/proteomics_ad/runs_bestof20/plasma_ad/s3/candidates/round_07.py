"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
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
    try:
        return np.array([str(m.get("batch", 0)) if isinstance(m, dict) else str(m) for m in meta],
                        dtype=object)
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_center(F_train, F_test, btr, bte):
    """Subtract per-batch, per-protein median of observed values (estimated on train)."""
    Ftr, Fte = F_train.copy(), F_test.copy()
    glob = F_train.median()
    for b in np.unique(np.concatenate([btr, bte])):
        mtr, mte = btr == b, bte == b
        if mtr.sum() >= 5:
            med = F_train.loc[mtr].median()
        elif mte.sum() >= 5:
            med = F_test.loc[mte].median()                 # label-free fallback for unseen batch
        else:
            continue
        shift = (med - glob).fillna(0.0)
        if mtr.any():
            Ftr.loc[mtr] = Ftr.loc[mtr] - shift
        if mte.any():
            Fte.loc[mte] = Fte.loc[mte] - shift
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    btr = _batches(ctx.get("meta_train"), len(F_train))
    bte = _batches(ctx.get("meta_test"), len(F_test))
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    F_train, F_test = _batch_center(F_train[keep], F_test[keep], btr, bte)
    fill = F_train.min() - 1.0                              # half-minimum on the log2 scale
    Xtr = F_train.fillna(fill).values
    Xte = F_test.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
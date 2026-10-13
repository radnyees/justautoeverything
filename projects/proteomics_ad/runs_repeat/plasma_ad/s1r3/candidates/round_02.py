"""log2 + per-sample median centring + half-minimum imputation + per-batch protein centring
+ batch-centred missingness indicators for mid-detection proteins + L2 logistic regression."""
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


def _batch_centre(Xtr, Xte, btr, bte, stat):
    """Remove per-batch offsets (stats from the training fold; unseen batch uses own label-free stats)."""
    g = stat(Xtr, axis=0)
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    for b in np.unique(np.concatenate([btr, bte])):
        mtr, mte = btr == b, bte == b
        if mtr.sum() >= 5:
            off = stat(Xtr[mtr], axis=0) - g
        elif mte.sum() >= 5:
            off = stat(Xte[mte], axis=0) - g
        else:
            off = 0.0
        Xtr_c[mtr] -= off
        Xte_c[mte] -= off
    return Xtr_c, Xte_c


def fit_predict(F_train, y_train, F_test, ctx):
    pres = F_train.notna().mean()
    keep = F_train.columns[pres >= 0.7]                     # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    Xtr_c, Xte_c = _batch_centre(Xtr, Xte, btr, bte, np.median)
    sc = StandardScaler().fit(Xtr_c)
    Ztr, Zte = sc.transform(Xtr_c), sc.transform(Xte_c)

    # detection indicators for proteins near the limit of detection (abundance-dependent missingness)
    mid = F_train.columns[(pres >= 0.2) & (pres < 0.7)]
    if len(mid) > 0:
        Mtr = F_train[mid].notna().values.astype(float)
        Mte = F_test[mid].notna().values.astype(float)
        Mtr_c, Mte_c = _batch_centre(Mtr, Mte, btr, bte, np.mean)
        msc = StandardScaler().fit(Mtr_c)
        Ztr = np.hstack([Ztr, 0.5 * msc.transform(Mtr_c)])
        Zte = np.hstack([Zte, 0.5 * msc.transform(Mte_c)])

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    return clf.predict_proba(Zte)[:, 1]
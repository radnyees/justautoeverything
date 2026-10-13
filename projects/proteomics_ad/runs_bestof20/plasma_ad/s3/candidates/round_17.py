"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
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
    arr = np.asarray(meta)
    return arr.astype(str) if arr.shape[0] == n else np.zeros(n, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Xtr = F_train[keep].copy()
    Xte = F_test[keep].copy()

    # per-batch, per-protein median centring (label-free, uses train+test of each batch)
    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    allX = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    allb = np.concatenate([btr, bte])
    for b in np.unique(allb):
        m = allb == b
        if m.sum() < 5:
            continue
        med = allX.loc[m].median(axis=0)
        allX.loc[m] = allX.loc[m] - med.fillna(0.0)
    Xtr = allX.iloc[:len(Xtr)].reset_index(drop=True)
    Xte = allX.iloc[len(Xtr):].reset_index(drop=True)

    fill = Xtr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Xtr.fillna(fill).values
    Xte = Xte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
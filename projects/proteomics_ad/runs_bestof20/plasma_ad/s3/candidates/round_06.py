"""log2 + per-sample median centring + half-minimum imputation + per-batch feature
standardization (train-batch statistics) + L2 logistic regression."""
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
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return pd.DataFrame(meta)["batch"].astype(str).values
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_standardize(Xtr, Xte, btr, bte, min_n=5):
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    g_mu = Xtr.mean(axis=0)
    g_sd = Xtr.std(axis=0) + 1e-6
    for b in np.unique(np.concatenate([btr, bte])):
        mtr = btr == b
        mte = bte == b
        if mtr.sum() >= min_n:
            mu = Xtr[mtr].mean(axis=0)
            sd = Xtr[mtr].std(axis=0) + 1e-6
        else:                                               # unseen / tiny batch: global stats
            mu, sd = g_mu, g_sd
        Xtr[mtr] = (Xtr[mtr] - mu) / sd
        Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _batch_standardize(Xtr, Xte, btr, bte)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
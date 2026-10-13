"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    try:
        if meta is None:
            return None
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns:
                return None
            b = meta["batch"].values
        elif isinstance(meta, pd.Series):
            b = meta.values
        else:
            b = np.asarray(meta)
        if len(b) != n:
            return None
        return np.asarray(b).astype(str)
    except Exception:
        return None


def _batch_centre(F_train, F_test, ctx):
    """Label-free per-batch, per-protein median centring (train+test pooled within batch)."""
    btr = _batches(ctx.get("meta_train"), len(F_train))
    bte = _batches(ctx.get("meta_test"), len(F_test))
    if btr is None or bte is None:
        return F_train, F_test
    A = pd.concat([F_train, F_test], axis=0, ignore_index=True)
    b = np.concatenate([btr, bte])
    out = A.copy()
    for lev in np.unique(b):
        m = b == lev
        if m.sum() < 5:
            continue
        sub = A.loc[m]
        med = sub.median(axis=0)
        cnt = sub.notna().sum(axis=0)
        med = med.where(cnt >= 3, 0.0).fillna(0.0)          # don't shift poorly observed proteins
        out.loc[m] = sub - med
    ntr = len(F_train)
    Ftr = out.iloc[:ntr]
    Fte = out.iloc[ntr:]
    Ftr.index = F_train.index
    Fte.index = F_test.index
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = pd.DataFrame(F_train)
    F_test = pd.DataFrame(F_test)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    F_train, F_test = _batch_centre(F_train[keep], F_test[keep], ctx)
    fill = F_train.min() - 1.0                              # half-minimum on the log2 scale
    Xtr = F_train.fillna(fill).values
    Xte = F_test.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
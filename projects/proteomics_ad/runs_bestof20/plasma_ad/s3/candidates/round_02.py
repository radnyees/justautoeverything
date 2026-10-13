"""log2 + per-sample median centring + per-batch per-protein median centring
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame):
        if "batch" not in meta.columns:
            return None
        b = meta["batch"]
        if b.index.equals(items.index) or set(items.index).issubset(set(b.index)):
            try:
                return b.reindex(items.index).values
            except Exception:
                pass
        if len(b) == len(items):
            return b.values
        return None
    try:
        b = np.asarray(meta["batch"])
        return b if len(b) == len(items) else None
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is None:
        return L
    b = pd.Series(b, index=L.index).astype(str)
    gmed = L.median(axis=0)
    out = L.copy()
    for lvl in b.unique():
        m = (b == lvl).values
        if m.sum() < 5:
            continue
        bmed = L.loc[m].median(axis=0)
        shift = (bmed - gmed).fillna(0.0)                   # proteins unseen in batch: no shift
        out.loc[m] = L.loc[m] - shift
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
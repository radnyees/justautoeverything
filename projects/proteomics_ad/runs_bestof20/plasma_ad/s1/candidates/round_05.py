"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta", None) if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            if meta.index.equals(items.index) or set(items.index) <= set(meta.index):
                b = meta.loc[items.index, "batch"]
            else:
                b = meta["batch"]
            b = np.asarray(b)
        else:
            b = np.asarray(pd.DataFrame(meta)["batch"])
        if len(b) != len(items):
            return None
        return b
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None:
        b = pd.Series(b, index=L.index).astype(str)
        gmed = L.median(axis=0)
        for lvl in b.unique():
            m = (b == lvl).values
            bmed = L.loc[m].median(axis=0)
            shift = (bmed - gmed).fillna(0.0)               # per-batch, per-protein offset (label-free)
            L.loc[m] = L.loc[m].sub(shift, axis=1)
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]
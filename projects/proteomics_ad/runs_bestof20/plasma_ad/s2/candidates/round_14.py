"""log2 + per-sample median centring + per-batch protein median harmonization
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            b = meta["batch"]
            if b.index.equals(items.index):
                return b.astype(str).values
            if len(b) == len(items):
                return b.astype(str).values
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_labels(items, ctx)
    if batch is None or len(np.unique(batch)) < 2:
        return L
    gmed = L.median(axis=0)                                # global per-protein median (label-free)
    out = L.copy()
    for b in np.unique(batch):
        m = batch == b
        sub = L.loc[m]
        bmed = sub.median(axis=0)
        enough = sub.notna().sum(axis=0) >= 3              # only correct well-observed proteins
        shift = (bmed - gmed).where(enough, 0.0).fillna(0.0)
        out.loc[m] = sub - shift
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
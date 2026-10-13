"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "batch" not in getattr(meta, "columns", []):
        return None
    if isinstance(meta, pd.DataFrame) and meta.index.equals(items.index):
        return meta["batch"].values
    if len(meta) == len(items):
        return np.asarray(meta["batch"])
    try:
        return meta.loc[items.index, "batch"].values
    except Exception:
        return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None:
        b = pd.Series(b, index=L.index).astype(str)
        if b.nunique() > 1:
            # label-free location harmonization: remove protein-specific batch offsets
            med = L.groupby(b).transform("median")
            L = L - med.fillna(0.0)
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
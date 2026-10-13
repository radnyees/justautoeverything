"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "batch" not in getattr(meta, "columns", []):
        return None
    b = meta["batch"]
    if b.index.equals(items.index) or set(items.index).issubset(set(b.index)):
        try:
            return b.loc[items.index].values
        except Exception:
            pass
    if len(b) == len(items):
        return np.asarray(b.values)
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _batch_vector(items, ctx)
    if batch is not None:
        L = L.copy()
        gmed = L.median(axis=0)                            # global per-protein median
        for bv in pd.unique(batch):
            m = batch == bv
            if m.sum() < 5:
                continue
            bmed = L.loc[m].median(axis=0)                 # per-batch per-protein median (observed values)
            shift = (bmed - gmed).fillna(0.0)
            L.loc[m] = L.loc[m].sub(shift, axis=1)         # align batch medians to global median
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
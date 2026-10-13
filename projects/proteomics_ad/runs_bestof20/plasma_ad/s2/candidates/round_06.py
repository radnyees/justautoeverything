"""log2 + per-sample median centring + per-batch protein median harmonization
+ half-minimum imputation + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _get_batch(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
        b = meta["batch"]
        if b.index.equals(items.index):
            return b.values
        try:
            rb = b.reindex(items.index)
            if rb.notna().all():
                return rb.values
        except Exception:
            pass
        if len(b) == len(items):
            return b.values
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    batch = _get_batch(items, ctx)
    if batch is not None and len(pd.unique(batch)) > 1:
        gmed = L.median(axis=0)                            # global per-protein median
        out = L.copy()
        for b in pd.unique(batch):
            m = batch == b
            if m.sum() < 5:
                continue
            bmed = L.loc[m].median(axis=0)
            shift = (bmed - gmed).fillna(0.0)              # label-free batch offset
            out.loc[m] = L.loc[m] - shift
        L = out
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
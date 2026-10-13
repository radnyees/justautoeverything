"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _get_batch(items, ctx):
    meta = ctx.get("meta", None) if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            col = "batch" if "batch" in meta.columns else ("site" if "site" in meta.columns else None)
            if col is None:
                return None
            s = meta[col]
            if s.index.equals(items.index):
                return s.values
            if set(items.index).issubset(set(s.index)):
                return s.loc[items.index].values
            if len(s) == len(items):
                return s.values
        elif isinstance(meta, dict) and "batch" in meta and len(meta["batch"]) == len(items):
            return np.asarray(meta["batch"])
    except Exception:
        return None
    return None


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _get_batch(items, ctx)
    if b is not None:
        b = pd.Series(b, index=L.index).astype(str)
        for lev in b.unique():
            m = (b == lev).values
            if m.sum() < 5:
                continue
            med = L.loc[m].median(axis=0)                  # per-protein, observed values only
            L.loc[m] = L.loc[m].sub(med.fillna(0.0), axis=1)
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
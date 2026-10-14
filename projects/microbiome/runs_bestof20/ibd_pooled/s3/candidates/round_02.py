"""Genus-level CLR with per-study median centering (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].values
        return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T             # samples x genera (relative abundance)
    # keep genera seen in at least 1% of samples to limit noise from singletons
    prev = (G > 0).mean(axis=0)
    G = G.loc[:, prev >= 0.01]
    # CLR with small pseudocount relative to typical abundance
    L = np.log(G.values + 1e-5)
    L = L - L.mean(axis=1, keepdims=True)
    # per-study median centering of each genus (label-free harmonization)
    studies = _study_labels(items, ctx)
    for s in pd.unique(studies):
        m = studies == s
        L[m] = L[m] - np.median(L[m], axis=0, keepdims=True)
    return pd.DataFrame(L, index=items.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
"""Genus-level CLR (depth-robust pseudocount) + per-study standardization (harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        s = meta["study"].reindex(items.index)
        if s.notna().all():
            return s.astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera (relative abundance)
    # keep genera present (>0) in at least 5% of samples
    G = G.loc[:, (G > 0).mean(axis=0) >= 0.05]
    L = np.log(G.values + 1e-5)                             # depth-independent pseudocount floor
    L = L - L.mean(axis=1, keepdims=True)                   # CLR across retained genera
    # per-study standardization (label-free harmonization of batch effects)
    studies = _study_labels(items, ctx)
    Z = np.zeros_like(L)
    for s in pd.unique(studies):
        m = studies == s
        mu = L[m].mean(axis=0)
        sd = L[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (L[m] - mu) / sd
    return pd.DataFrame(Z, index=items.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
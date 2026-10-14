"""Genus-level log relative abundance, per-study robust standardization (harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta["study"].reindex(items.index).fillna("NA").astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G.values + 1e-5)
    study = _study_labels(items, ctx)
    Z = np.empty_like(L)
    for s in pd.unique(study):
        m = study == s
        block = L[m]
        med = np.median(block, axis=0)
        q75, q25 = np.percentile(block, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = block.std(axis=0)
        scale = np.where(scale > 1e-6, scale, np.where(sd > 1e-6, sd, 1.0))
        Z[m] = (block - med) / scale
    Z = np.clip(Z, -10, 10)
    return pd.DataFrame(Z, index=G.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
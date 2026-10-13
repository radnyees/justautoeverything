"""Genus-level log relative abundance, z-scored within each study (label-free batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta.loc[items.index, "study"]
        if len(s) == len(items):
            return s.astype(str).values
    except Exception:
        pass
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    counts = items.astype(float)
    depth = counts.sum(axis=1).clip(lower=1)
    rel = counts.div(depth, axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    # depth-aware pseudocount: half a read at each sample's depth
    pc = (0.5 / depth.values)[:, None]
    L = np.log10(G.values + pc)
    # per-study standardization (label-free harmonization)
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
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    return rf.predict_proba(Xte)[:, 1]
"""Genus-level relative abundance, log-transformed and z-scored within each study
(batch harmonization), followed by a random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    s = meta["study"]
    if s.index.equals(items.index):
        return s.values
    try:
        r = s.reindex(items.index)
        if r.notna().all():
            return r.values
    except Exception:
        pass
    if len(s) == len(items):
        return s.values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G.values + 1e-5)
    study = _study_labels(items, ctx)
    Z = np.zeros_like(L)
    for s in pd.unique(study):
        m = study == s
        mu = L[m].mean(axis=0)
        sd = L[m].std(axis=0)
        sd[sd < 1e-8] = 1.0                                  # genus absent in study -> 0
        Z[m] = (L[m] - mu) / sd
    return pd.DataFrame(Z, index=G.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
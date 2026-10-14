"""Genus-level relative abundance, harmonized by per-study percentile ranks + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None) if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        if meta.index.isin(items.index).all() or items.index.isin(meta.index).all():
            s = meta["study"].reindex(items.index)
            if s.notna().all():
                return s.astype(str).values
    except Exception:
        pass
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    # per-study percentile ranks (label-free harmonization of depth/protocol shifts)
    study = _study_labels(G, ctx)
    out = np.zeros(G.shape, dtype=float)
    vals = G.values
    for s in pd.unique(study):
        m = study == s
        sub = pd.DataFrame(vals[m])
        out[m] = sub.rank(axis=0, method="average", pct=True).values
    return pd.DataFrame(out, index=G.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
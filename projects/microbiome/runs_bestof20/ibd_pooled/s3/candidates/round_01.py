"""Genus-level relative abundance, harmonized by within-study percentile ranks, + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=int)
    try:
        s = meta.loc[items.index, "study"]
        if len(s) == len(items):
            return s.astype(str).values
    except Exception:
        pass
    return meta["study"].astype(str).values[: len(items)]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T                # samples x genera
    study = _study_labels(items, ctx)
    # per-study percentile rank of each genus: removes study-specific depth/protocol scaling
    R = G.groupby(study).rank(pct=True, method="average")
    return R.fillna(0.5)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
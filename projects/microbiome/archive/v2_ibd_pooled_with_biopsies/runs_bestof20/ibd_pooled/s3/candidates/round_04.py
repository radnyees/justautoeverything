"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in meta:
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.values
    except Exception:
        return meta["study"].values[: len(items)]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    study = _study_labels(items, ctx)
    out = G.copy().astype(float)
    for s in pd.unique(study):
        m = study == s
        # within-study percentile rank (ties -> average), removes depth/protocol shifts
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
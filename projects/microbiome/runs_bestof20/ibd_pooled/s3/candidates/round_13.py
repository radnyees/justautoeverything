"""Genus-level relative abundance, percentile-ranked within each study (label-free batch
harmonization), + random forest."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    vals = G.values.astype(float)
    out = np.zeros_like(vals)
    studies = _study_labels(items, ctx)
    for s in pd.unique(studies):
        m = studies == s
        n = m.sum()
        sub = vals[m]
        # percentile rank per genus within the study (ties, e.g. zeros, share a rank)
        out[m] = np.apply_along_axis(lambda v: rankdata(v, method="average"), 0, sub) / max(n, 1)
    return pd.DataFrame(out, index=G.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
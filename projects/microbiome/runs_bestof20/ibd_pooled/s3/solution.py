"""Genus-level relative abundance, harmonized by within-study percentile ranks + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index):
        return meta["study"].values
    if items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].values
    if len(meta) == len(items):
        return meta["study"].values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    # Label-free per-study harmonization: percentile rank of each genus within its study
    study = pd.Series(_study_labels(G, ctx), index=G.index).astype(str)
    R = G.groupby(study.values, group_keys=False).apply(
        lambda d: d.rank(axis=0, method="average", pct=True))
    R = R.loc[G.index]
    # genera absent in a whole study get a neutral constant (rank of ties = 1.0); set to 0 instead
    zero = G.groupby(study.values).transform(lambda s: float((s > 0).any()))
    R = R * zero.loc[G.index].values
    return R


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
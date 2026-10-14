"""Genus-level relative abundance, percentile-ranked within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=items.index)
    if meta.index.equals(items.index) or items.index.isin(meta.index).all():
        s = meta["study"].reindex(items.index)
    elif len(meta) == len(items):
        s = pd.Series(meta["study"].values, index=items.index)
    else:
        return pd.Series("all", index=items.index)
    return s.fillna("unknown").astype(str)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    study = _study_labels(items, ctx)
    # within-study percentile rank: removes study-specific depth/protocol scale shifts
    R = G.groupby(study.values).rank(pct=True, method="average")
    R = R.reindex(index=G.index, columns=G.columns).fillna(0.5)
    return R.astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
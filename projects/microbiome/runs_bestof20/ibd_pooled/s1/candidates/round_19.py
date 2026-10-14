"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_of(index, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=index)
    s = meta["study"].reindex(index)
    return s.fillna("unknown").astype(str)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    study = _study_of(G.index, ctx)
    # within-study percentile ranks (label-free harmonization of batch offsets/depth effects)
    R = G.groupby(study.values).rank(method="average", pct=True)
    R = R.loc[G.index]
    # genera entirely absent in a study -> rank ties; set to a neutral constant 0
    absent = G.groupby(study.values).transform(lambda x: (x > 0).any())
    R = R.where(absent.astype(bool), 0.0)
    return R.astype(float)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
"""Genus-level relative abundance, percentile-normalized within study (label-free batch harmonization)
+ random forest trained with study x class balanced sample weights (each study contributes equally,
with 50/50 case/control balance, so the model learns within-study contrasts, not study prevalence)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta["study"].reindex(items.index).astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return None


def _train_studies(F_train, ctx):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    idx = getattr(F_train, "index", None)
    if idx is not None and set(idx).issubset(set(meta.index)):
        return meta["study"].reindex(idx).astype(str).values
    if len(meta) == len(F_train):
        return meta["study"].astype(str).values
    return None


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    if studies is None:
        return G.rank(pct=True)
    out = G.copy()
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, pct=True, method="average").values
    return out


def _balanced_weights(y, studies):
    y = np.asarray(y)
    if studies is None:
        studies = np.zeros(len(y), dtype=object)
    w = np.zeros(len(y), dtype=float)
    for s in pd.unique(studies):
        ms = studies == s
        classes = np.unique(y[ms])
        for c in classes:
            m = ms & (y == c)
            w[m] = 1.0 / (m.sum() * len(classes))           # each study sums to 1, split evenly by class
    return w * len(y) / w.sum()


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    w = _balanced_weights(y_train, _train_studies(F_train, ctx))
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train), sample_weight=w)
    return rf.predict_proba(Xte)[:, 1]
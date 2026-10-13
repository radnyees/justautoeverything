"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].values
        return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    gen = rel.T.groupby(np.array(genus)).sum().T            # samples x genera
    studies = _study_labels(items, ctx)
    out = pd.DataFrame(0.5, index=gen.index, columns=gen.columns, dtype=float)
    for s in pd.unique(studies):
        mask = studies == s
        sub = gen.loc[mask]
        # within-study percentile ranks (ties, e.g. zeros, get their average rank)
        out.loc[mask] = sub.rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
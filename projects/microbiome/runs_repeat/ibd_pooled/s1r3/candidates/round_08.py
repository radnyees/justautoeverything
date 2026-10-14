"""Lineage-aware phylum..genus relative abundance, percentile-ranked within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
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


def _lineage(col, depth):
    # Full lineage prefix up to `depth` levels (k,p = 2; c = 3; o = 4; f = 5; g = 6), so that
    # unannotated levels (e.g. 'g__') stay distinct per parent taxon instead of being lumped together.
    parts = col.split(";")
    parts = parts + [""] * max(0, depth - len(parts))
    return ";".join(parts[:depth])


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    blocks = []
    for depth in (2, 3, 4, 5, 6):                           # phylum, class, order, family, genus
        keys = np.array([_lineage(c, depth) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    studies = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
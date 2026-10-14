"""Lineage-aware family+genus relative abundance, prevalence-filtered (per study), percentile-ranked within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MIN_PREV = 0.10   # keep taxa present in >= 10% of samples of at least one study


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
    # Full lineage prefix up to `depth` levels (k,p,c,o,f = 5; +g = 6), so that
    # unannotated genera ('g__') stay distinct per family instead of being lumped together.
    parts = col.split(";")
    parts = parts + [""] * max(0, depth - len(parts))
    return ";".join(parts[:depth])


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    blocks = []
    for depth in (5, 6):                                    # family, genus
        keys = np.array([_lineage(c, depth) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    studies = _study_labels(G, ctx)
    # Label-free prevalence filter: rare taxa become near-constant ranks that only
    # dilute the forest's feature subsampling; keep taxa common in at least one study.
    present = (G.values > 0)
    max_prev = np.zeros(G.shape[1])
    for s in pd.unique(studies):
        m = studies == s
        max_prev = np.maximum(max_prev, present[m].mean(axis=0))
    G = G.loc[:, max_prev >= MIN_PREV]
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
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
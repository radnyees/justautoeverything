"""Lineage-aware phylum..genus robust-CLR (log-ratio to non-zero geometric mean), percentile-ranked within study + random forest."""
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


def _robust_clr(C):
    # rCLR: log(count) minus the mean log over the sample's non-zero taxa at this level.
    # No pseudocount (depth-independent); zeros become one tied minimum value.
    X = C.values.astype(float)
    nz = X > 0
    L = np.where(nz, np.log(np.where(nz, X, 1.0)), 0.0)
    n = nz.sum(axis=1)
    gm = np.where(n > 0, L.sum(axis=1) / np.maximum(n, 1), 0.0)
    R = np.where(nz, L - gm[:, None], -1e6)
    return pd.DataFrame(R, index=C.index, columns=C.columns)


def featurize(items, ctx):
    blocks = []
    for depth in (2, 3, 4, 5, 6):                           # phylum, class, order, family, genus
        keys = np.array([_lineage(c, depth) for c in items.columns])
        C = items.T.groupby(keys).sum().T
        blocks.append(_robust_clr(C))
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
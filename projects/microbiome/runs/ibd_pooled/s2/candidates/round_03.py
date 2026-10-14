"""Multi-rank (phylum..genus, full-lineage-prefix) relative abundance, percentile-normalized within study + random forest."""
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


def _lineage_prefix(col, k):
    parts = [p.strip() for p in col.split(";")]
    parts = parts + [""] * max(0, k + 1 - len(parts))
    return ";".join(parts[: k + 1])


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    blocks = []
    for k in (1, 2, 3, 4, 5):                               # phylum, class, order, family, genus
        keys = np.array([_lineage_prefix(c, k) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    studies = _study_labels(items, ctx)
    if studies is None:
        return G.rank(pct=True).astype(np.float32)
    out = G.copy()
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, pct=True, method="average").values
    return out.astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
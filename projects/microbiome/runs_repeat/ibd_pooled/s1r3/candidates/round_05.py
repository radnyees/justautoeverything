"""Lineage-aware family+genus relative abundance, percentile-ranked within study (batch harmonization);
rank-averaged ensemble of random forest + inner-CV-tuned L2 logistic regression."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


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
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    studies = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def _inner_splits(y, groups, n=5):
    if groups is not None and len(np.unique(groups)) >= n:
        return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # Linear model: additive, monotone taxon shifts (e.g. Faecalibacterium loss, Enterobacteriaceae gain).
    sc = StandardScaler().fit(Xtr)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None else None
    lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=_inner_splits(y, groups),
                              penalty="l2", scoring="roc_auc", max_iter=5000, n_jobs=-1)
    lr.fit(sc.transform(Xtr), y)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]

    # Rank-average (scale-free combination of the two models).
    return (rankdata(p_rf) + rankdata(p_lr)) / (2.0 * len(p_rf))
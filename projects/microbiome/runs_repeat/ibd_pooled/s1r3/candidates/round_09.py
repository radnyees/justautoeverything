"""Lineage-aware phylum..genus relative abundance, percentile-ranked within study (batch harmonization)
+ random forest whose max_features is chosen by out-of-bag within-study AUROC on the training fold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def _train_studies(F_train, ctx):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(F_train), dtype=object)
    if F_train.index.isin(meta.index).all():
        return meta.loc[F_train.index, "study"].astype(str).values
    if len(meta) == len(F_train):
        return meta["study"].astype(str).values
    return np.zeros(len(F_train), dtype=object)


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


def _within_study_auc(y, p, studies):
    aucs, ws = [], []
    for s in pd.unique(studies):
        m = (studies == s) & np.isfinite(p)
        if m.sum() < 4 or len(np.unique(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], p[m]))
        ws.append(m.sum())
    if not aucs:
        m = np.isfinite(p)
        return roc_auc_score(y[m], p[m]) if len(np.unique(y[m])) > 1 else 0.5
    return float(np.average(aucs, weights=ws))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train).astype(int)
    studies = _train_studies(F_train, ctx)
    best_rf, best_auc = None, -np.inf
    for mf in ("sqrt", 0.05, 0.15, 0.3):
        rf = RandomForestClassifier(n_estimators=500, max_features=mf, oob_score=True,
                                    random_state=0, n_jobs=-1)
        rf.fit(Xtr, y)
        oob = rf.oob_decision_function_[:, 1]
        auc = _within_study_auc(y, oob, studies)
        if auc > best_auc + 1e-4:                           # ties keep the simpler/default choice
            best_rf, best_auc = rf, auc
    return best_rf.predict_proba(Xte)[:, 1]
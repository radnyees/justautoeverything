"""Genus + family relative abundance with within-study percentile-rank (batch-corrected) views + large random forest."""
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
        return s.astype(str).values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].astype(str).values
        return np.zeros(len(items), dtype=object)


def _level_name(col, level):
    parts = col.split(";")
    # Keep the full lineage up to this level so unassigned names (e.g. "g__") stay distinct per parent
    return ";".join(parts[: level + 1]) if len(parts) > level else ";".join(parts)


def _aggregate(rel, level):
    names = np.array([_level_name(c, level) for c in rel.columns])
    return rel.T.groupby(names).sum().T


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    Fm = _aggregate(rel, 4)                                 # samples x families (full lineage names)
    Fm.columns = ["fam_" + str(c) for c in Fm.columns]
    study = _study_labels(items, ctx)
    # Within-study percentile ranks (label-free batch correction across studies)
    RG = G.groupby(study).rank(method="average", pct=True).fillna(0.5)
    RG.columns = ["rk_" + str(c) for c in G.columns]
    RF = Fm.groupby(study).rank(method="average", pct=True).fillna(0.5)
    RF.columns = ["rk_" + str(c) for c in Fm.columns]
    return pd.concat([G, RG, Fm, RF], axis=1)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    # More trees -> finer-grained, lower-variance vote fractions for ranking (AUROC)
    rf = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + random forest with inner-CV-tuned split/leaf settings."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
GRID = [(mf, leaf) for mf in ["sqrt", 0.25] for leaf in [1, 3, 6]]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-5)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_logsum"] = np.log10(1e-5)
        G["marker_count"] = 0.0
    return G


def _select_cols(F):
    genus_cols = [c for c in F.columns if c not in MARKER_FEATS]
    prev = (F[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    return keep + [c for c in MARKER_FEATS if c in F.columns]


def _rf(mf, leaf, n=500):
    return RandomForestClassifier(n_estimators=n, max_features=mf, min_samples_leaf=leaf,
                                  random_state=0, n_jobs=-1)


def _tune(F_train, y, groups):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    splits = list(cv.split(F_train, y, groups))
    best, best_score = GRID[0], -np.inf
    for mf, leaf in GRID:
        aucs = []
        for tr, va in splits:
            Ftr, Fva = F_train.iloc[tr], F_train.iloc[va]
            cols = _select_cols(Ftr)
            m = _rf(mf, leaf, n=300).fit(Ftr[cols].values, y[tr])
            p = m.predict_proba(Fva.reindex(columns=cols, fill_value=0).values)[:, 1]
            if len(np.unique(y[va])) > 1:
                aucs.append(roc_auc_score(y[va], p))
        s = np.mean(aucs) if aucs else -np.inf
        if s > best_score:
            best, best_score = (mf, leaf), s
    return best


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    mf, leaf = _tune(F_train, y, groups)
    cols = _select_cols(F_train)
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = _rf(mf, leaf, n=1000)
    rf.fit(Xtr, y)
    return rf.predict_proba(Xte)[:, 1]
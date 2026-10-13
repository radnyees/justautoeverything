"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + RF / CatBoost probability-averaged ensemble."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
N_TREES = 3000                                              # more trees -> less Monte Carlo noise in ranking
RF_WEIGHT = 0.5                                             # weight of RF in the RF/CatBoost average


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


def fit_predict(F_train, y_train, F_test, ctx):
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    cols = keep + [c for c in MARKER_FEATS if c in F_train.columns]
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train).astype(int)

    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    cb = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=4, l2_leaf_reg=5,
                            random_seed=0, verbose=0, thread_count=-1, allow_writing_files=False)
    cb.fit(Xtr, y)
    p_cb = cb.predict_proba(Xte)[:, 1]

    return RF_WEIGHT * p_rf + (1 - RF_WEIGHT) * p_cb
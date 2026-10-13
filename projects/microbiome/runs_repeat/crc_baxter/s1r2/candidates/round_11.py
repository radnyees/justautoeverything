"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
(depth-normalized expected detection count) + training-fold prevalence filtering
+ large (low-variance) random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
N_TREES = 3000                                              # more trees -> less Monte Carlo noise in ranking
RAREFY_DEPTH = 5000                                         # standard depth for expected marker detection


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-5)
        # expected number of markers detected if every sample were rarefied to RAREFY_DEPTH reads
        p = np.clip(M.values, 0.0, 1.0 - 1e-12)
        p_det = 1.0 - np.exp(RAREFY_DEPTH * np.log1p(-p))
        G["marker_count"] = p_det.sum(axis=1)
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
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
+ pathogen-vs-butyrate-producer log-ratio balance
+ training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
# Butyrate-producing commensals consistently depleted in CRC stool
BUTYRATE_PRODUCERS = ["Faecalibacterium", "Roseburia", "Butyricicoccus", "Coprococcus",
                      "Lachnospira", "Pseudobutyrivibrio"]
MARKER_FEATS = ["marker_logsum", "marker_count", "marker_balance"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
EPS = 1e-5


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    bp_cols = [c for c in G.columns if names[c] in BUTYRATE_PRODUCERS]
    mk_sum = G[mk_cols].sum(axis=1) if mk_cols else pd.Series(0.0, index=G.index)
    bp_sum = G[bp_cols].sum(axis=1) if bp_cols else pd.Series(0.0, index=G.index)
    G["marker_logsum"] = np.log10(mk_sum + EPS)
    G["marker_count"] = (G[mk_cols] > 0).sum(axis=1).astype(float) if mk_cols else 0.0
    G["marker_balance"] = np.log10(mk_sum + EPS) - np.log10(bp_sum + EPS)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    cols = keep + [c for c in MARKER_FEATS if c in F_train.columns]
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
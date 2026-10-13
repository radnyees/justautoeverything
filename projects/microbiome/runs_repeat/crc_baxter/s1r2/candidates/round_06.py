"""Genus-level relative abundance (unclassified genera resolved to their deepest named
taxonomic level) + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0


def _genus_label(col):
    """Genus name if assigned; otherwise deepest named rank + 'g__unclassified'."""
    parts = [p.strip() for p in col.split(";")]
    ranks = parts[:6]                                       # k, p, c, o, f, g
    if len(ranks) > 5 and ranks[5] not in ("g__", ""):
        return ranks[5]
    for p in reversed(ranks[:5]):
        if len(p) > 3:                                      # e.g. 'f__Lachnospiraceae'
            return p + ";g__unclassified"
    return "g__unclassified"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [_genus_label(c) for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.replace("g__", "") for g in G.columns], index=G.columns)
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
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
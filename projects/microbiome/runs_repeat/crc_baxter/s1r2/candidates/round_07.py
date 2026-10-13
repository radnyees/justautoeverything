"""Genus- + family-level relative abundance + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with taxon > 0


def _level(cols, idx, empty):
    out = []
    for c in cols:
        parts = c.split(";")
        out.append(parts[idx].strip() if len(parts) > idx else empty)
    return np.array(out)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    G = rel.T.groupby(_level(rel.columns, 5, "g__")).sum().T      # samples x genera
    Fam = rel.T.groupby(_level(rel.columns, 4, "f__")).sum().T    # samples x families
    Fam = Fam[[c for c in Fam.columns if c != "f__"]]             # drop unassigned-family bucket
    Fam.columns = ["FAM|" + c for c in Fam.columns]
    names = pd.Series([g.replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    X = pd.concat([G, Fam], axis=1)
    if mk_cols:
        M = G[mk_cols]
        X["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-5)
        X["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        X["marker_logsum"] = np.log10(1e-5)
        X["marker_count"] = 0.0
    return X


def fit_predict(F_train, y_train, F_test, ctx):
    tax_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[tax_cols] > 0).mean(axis=0)
    keep = [c for c in tax_cols if prev[c] >= MIN_PREVALENCE]
    cols = keep + [c for c in MARKER_FEATS if c in F_train.columns]
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
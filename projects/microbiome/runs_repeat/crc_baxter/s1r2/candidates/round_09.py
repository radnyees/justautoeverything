"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
(core panel + extended oral panel) + training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
# Extended oral / anaerobic-pathobiont panel from CRC meta-analyses (Wirbel 2019, Thomas 2019)
EXT_MARKERS = CRC_MARKERS + ["Solobacterium", "Peptoniphilus", "Anaerococcus", "Campylobacter"]
MARKER_FEATS = ["marker_logsum", "marker_count", "ext_logsum", "ext_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0


def _panel(G, names, panel, prefix):
    cols = [c for c in G.columns if names.get(c, "") in panel]
    if cols:
        M = G[cols]
        return np.log10(M.sum(axis=1) + 1e-5), (M > 0).sum(axis=1).astype(float)
    n = len(G)
    return pd.Series(np.log10(1e-5), index=G.index), pd.Series(np.zeros(n), index=G.index)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = {g: g.strip().replace("g__", "").strip("[]") for g in G.columns}
    G = G.copy()
    G["marker_logsum"], G["marker_count"] = _panel(G, names, CRC_MARKERS, "marker")
    G["ext_logsum"], G["ext_count"] = _panel(G, names, EXT_MARKERS, "ext")
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
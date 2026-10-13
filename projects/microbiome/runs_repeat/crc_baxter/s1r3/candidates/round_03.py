"""Genus-level relative abundance + aggregate CRC oral-pathobiont markers + pathobiont/butyrate-producer
log-ratio balances + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-origin pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CORE_MARKERS = ["g__Fusobacterium", "g__Porphyromonas", "g__Parvimonas", "g__Peptostreptococcus"]
EXT_MARKERS = CORE_MARKERS + ["g__Gemella", "g__Solobacterium", "g__Prevotella",
                              "g__Campylobacter", "g__Dialister", "g__Filifactor"]
# Butyrate-producing commensals depleted in CRC
BUTYRATE = ["g__Faecalibacterium", "g__Roseburia", "g__Coprococcus", "g__Butyricicoccus",
            "g__Anaerostipes", "g__Lachnospira", "g__Eubacterium", "g__Pseudobutyrivibrio",
            "g__Butyrivibrio", "g__Subdoligranulum"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera

    def col(name):
        return G[name] if name in G.columns else pd.Series(0.0, index=G.index)

    core = sum(col(g) for g in CORE_MARKERS)
    ext = sum(col(g) for g in EXT_MARKERS if g != "g__Prevotella")
    but = sum(col(g) for g in BUTYRATE)
    G = G.copy()
    G["MK_core_sum_log"] = np.log10(core + 1e-6)
    G["MK_ext_sum_log"] = np.log10(ext + 1e-6)
    G["MK_core_npresent"] = sum((col(g) > 0).astype(float) for g in CORE_MARKERS)
    G["MK_ext_npresent"] = sum((col(g) > 0).astype(float) for g in EXT_MARKERS)
    # log-ratio balances: pathobionts vs butyrate producers (compositional, depth-independent)
    pc = 1e-5
    G["BAL_core_vs_but"] = np.log10(core + pc) - np.log10(but + pc)
    G["BAL_ext_vs_but"] = np.log10(ext + pc) - np.log10(but + pc)
    G["MK_but_sum_log"] = np.log10(but + 1e-6)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
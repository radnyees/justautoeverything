"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
+ OTU-level alpha diversity (Shannon, inverse Simpson, rarefied richness)
+ training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
DIV_FEATS = ["div_shannon", "div_invsimpson", "div_richness_rare", "div_genus_shannon"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
EXTRA_FEATS = MARKER_FEATS + DIV_FEATS
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
RARE_DEPTH = 5000                                           # fixed rarefaction depth for richness


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log(P), 0.0)
    return -L.sum(axis=1)


def _alpha_diversity(items):
    C = items.values.astype(np.int64)
    depth = C.sum(axis=1)
    P = C / np.clip(depth, 1, None)[:, None]
    shannon = _shannon(P)
    invsimp = 1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None)
    rng = np.random.default_rng(0)
    rich = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        row = C[i]
        if depth[i] > RARE_DEPTH:
            nz = np.flatnonzero(row)
            sub = rng.multivariate_hypergeometric(row[nz], RARE_DEPTH, method="marginals")
            rich[i] = float((sub > 0).sum())
        else:
            rich[i] = float((row > 0).sum())
    return shannon, invsimp, rich


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    genus_shannon = _shannon(G.values)
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-5)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_logsum"] = np.log10(1e-5)
        G["marker_count"] = 0.0
    shannon, invsimp, rich = _alpha_diversity(items)
    G["div_shannon"] = shannon
    G["div_invsimpson"] = invsimp
    G["div_richness_rare"] = rich
    G["div_genus_shannon"] = genus_shannon
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    genus_cols = [c for c in F_train.columns if c not in EXTRA_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    cols = keep + [c for c in EXTRA_FEATS if c in F_train.columns]
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
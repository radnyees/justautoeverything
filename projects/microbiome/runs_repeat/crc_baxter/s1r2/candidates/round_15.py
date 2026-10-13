"""Genus + family relative abundance (log-scaled) + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + training-fold univariate (Mann-Whitney) selection over the
pooled genus/family candidate set + large extremely-randomized-trees ensemble."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with taxon > 0
TOP_K = 40                                                  # taxa kept after univariate case/control test
N_TREES = 3000                                              # more trees -> less Monte Carlo noise in ranking
LOG_FLOOR = 1e-5                                            # pseudo-abundance below typical detection limit


def _level(c, i, prefix):
    parts = c.split(";")
    return parts[i].strip() if len(parts) > i else prefix


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [_level(c, 5, "g__") for c in rel.columns]
    family = [_level(c, 4, "f__") for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    Fam = rel.T.groupby(np.array(family)).sum().T           # samples x families
    Fam.columns = ["fam:" + c for c in Fam.columns]
    names = pd.Series([g.replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    out = pd.concat([G, Fam], axis=1)
    if mk_cols:
        M = G[mk_cols]
        out["marker_logsum"] = np.log10(M.sum(axis=1) + LOG_FLOOR)
        out["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        out["marker_logsum"] = np.log10(LOG_FLOOR)
        out["marker_count"] = 0.0
    return out


def _select_taxa(X, y, cols, k):
    """Rank taxa by two-sided Mann-Whitney p-value (cases vs controls) on the training fold."""
    y = np.asarray(y)
    if len(cols) <= k or len(np.unique(y)) < 2:
        return cols
    pos, neg = X[y == 1], X[y == 0]
    pvals = np.ones(len(cols))
    for j in range(len(cols)):
        a, b = pos[:, j], neg[:, j]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        pvals[j] = mannwhitneyu(a, b, alternative="two-sided").pvalue
    order = np.argsort(pvals, kind="stable")[:k]
    return [cols[i] for i in sorted(order)]


def fit_predict(F_train, y_train, F_test, ctx):
    taxa_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[taxa_cols] > 0).mean(axis=0)
    keep = [c for c in taxa_cols if prev[c] >= MIN_PREVALENCE]
    keep = _select_taxa(np.log10(F_train[keep].values + LOG_FLOOR), y_train, keep, TOP_K)
    mk = [c for c in MARKER_FEATS if c in F_train.columns]
    # Log-scale abundances so random split thresholds spread over the informative range
    Xtr = np.hstack([np.log10(F_train[keep].values + LOG_FLOOR), F_train[mk].values])
    Fte = F_test.reindex(columns=keep + mk, fill_value=0)
    Xte = np.hstack([np.log10(Fte[keep].values + LOG_FLOOR), Fte[mk].values])
    et = ExtraTreesClassifier(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    et.fit(Xtr, y_train)
    return et.predict_proba(Xte)[:, 1]
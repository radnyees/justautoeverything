"""Genus-level relative abundance (log-scaled) + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + subsample-bagged univariate (Mann-Whitney) genus selection
+ extremely-randomized-trees ensemble averaged over selection bags."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
TOP_K = 40                                                  # genera kept after univariate case/control test
N_BAGS = 10                                                 # selection subsamples (stabilize the top-K cutoff)
SUBSAMPLE = 0.8                                             # fraction of training subjects per selection bag
N_TREES = 3000                                              # total trees, split evenly over bags
LOG_FLOOR = 1e-5                                            # pseudo-abundance below typical detection limit


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + LOG_FLOOR)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_logsum"] = np.log10(LOG_FLOOR)
        G["marker_count"] = 0.0
    return G


def _select_genera(X, y, cols, k):
    """Rank genera by two-sided Mann-Whitney p-value (cases vs controls) on the given samples."""
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
    y = np.asarray(y_train)
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    mk = [c for c in MARKER_FEATS if c in F_train.columns]
    # Log-scale genus abundances so random split thresholds spread over the informative range
    Ltr = np.log10(F_train[keep].values + LOG_FLOOR)
    Fte = F_test.reindex(columns=keep + mk, fill_value=0)
    Lte = np.log10(Fte[keep].values + LOG_FLOOR)
    Mtr, Mte = F_train[mk].values, Fte[mk].values

    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None else np.arange(len(y))
    uniq = np.unique(groups)
    n_sub = max(2, int(round(SUBSAMPLE * len(uniq))))
    trees_per_bag = max(1, N_TREES // N_BAGS)

    pred = np.zeros(len(F_test))
    for b in range(N_BAGS):
        rng = np.random.RandomState(b)
        sub_groups = rng.choice(uniq, size=n_sub, replace=False)
        idx = np.isin(groups, sub_groups)
        sel = _select_genera(Ltr[idx], y[idx], list(range(len(keep))), TOP_K)
        Xtr = np.hstack([Ltr[:, sel], Mtr])
        Xte = np.hstack([Lte[:, sel], Mte])
        et = ExtraTreesClassifier(n_estimators=trees_per_bag, random_state=b, n_jobs=-1)
        et.fit(Xtr, y)
        pred += et.predict_proba(Xte)[:, 1]
    return pred / N_BAGS
"""Genus-level relative abundance + aggregate CRC oral-pathobiont marker features + depth-normalized
alpha diversity; rank-averaged ensemble of a random forest (genera + markers + diversity) and an L2
logistic regression on multi-resolution (genus + family) log-abundances."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Oral-origin pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CORE_MARKERS = ["g__Fusobacterium", "g__Porphyromonas", "g__Parvimonas", "g__Peptostreptococcus"]
EXT_MARKERS = CORE_MARKERS + ["g__Gemella", "g__Solobacterium", "g__Prevotella",
                              "g__Campylobacter", "g__Dialister", "g__Filifactor"]
PSEUDO = 5e-5          # ~half the smallest detectable relative abundance at typical MiSeq depth
MIN_PREV = 0.10        # prevalence filter for the linear model only (label-free, per training fold)
FAM_PREFIX = "FAM|"    # family-level aggregates, used by the linear model only
DENSE_PREFIXES = ("MK_", "DIV_")   # engineered dense features, never prevalence-filtered or logged
RARE_DEPTH = 2000      # common depth for rarefied (Hurlbert) richness


def _level(c, i, empty):
    parts = c.split(";")
    return parts[i] if len(parts) > i else empty


def _rarefied_richness(counts, n=RARE_DEPTH):
    """Expected number of taxa in a random subsample of n reads (Hurlbert 1971), per sample."""
    counts = np.asarray(counts, dtype=float)
    out = np.empty(counts.shape[0])
    for k, row in enumerate(counts):
        c = row[row > 0]
        N = c.sum()
        if N <= n:
            out[k] = len(c)
            continue
        rest = N - c
        ok = rest >= n
        logp = np.zeros_like(c)
        logp[ok] = (gammaln(rest[ok] + 1) - gammaln(rest[ok] - n + 1)
                    - gammaln(N + 1) + gammaln(N - n + 1))
        p_absent = np.where(ok, np.exp(logp), 0.0)
        out[k] = np.sum(1.0 - p_absent)
    return out


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [_level(c, 5, "g__") for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera
    family = [FAM_PREFIX + _level(c, 4, "f__") for c in rel.columns]
    Fm = rel.T.groupby(np.array(family)).sum().T        # samples x families

    def col(name):
        return G[name] if name in G.columns else pd.Series(0.0, index=G.index)

    core = sum(col(g) for g in CORE_MARKERS)
    ext = sum(col(g) for g in EXT_MARKERS if g != "g__Prevotella")
    G = G.copy()
    G["MK_core_sum_log"] = np.log10(core + 1e-6)
    G["MK_ext_sum_log"] = np.log10(ext + 1e-6)
    G["MK_core_npresent"] = sum((col(g) > 0).astype(float) for g in CORE_MARKERS)
    G["MK_ext_npresent"] = sum((col(g) > 0).astype(float) for g in EXT_MARKERS)

    # Alpha diversity: evenness (Shannon) and depth-normalized richness at OTU and genus level
    P = rel.values
    with np.errstate(divide="ignore", invalid="ignore"):
        G["DIV_shannon"] = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    G["DIV_rich_otu"] = _rarefied_richness(items.values)
    genus_counts = items.T.groupby(np.array(genus)).sum().T
    G["DIV_rich_genus"] = _rarefied_richness(genus_counts.values)
    return pd.concat([G, Fm], axis=1)


def _linear_scores(F_train, y_train, F_test, ctx):
    taxa = [c for c in F_train.columns if not c.startswith(DENSE_PREFIXES)]   # genera + families
    mk = [c for c in F_train.columns if c.startswith(DENSE_PREFIXES)]
    prev = (F_train[taxa] > 0).mean(axis=0)
    keep = [g for g in taxa if prev[g] >= MIN_PREV]
    Xtr = np.hstack([np.log10(F_train[keep].values + PSEUDO), F_train[mk].values])
    Xte = np.hstack([np.log10(F_test.reindex(columns=keep, fill_value=0).values + PSEUDO),
                     F_test.reindex(columns=mk, fill_value=0).values])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    cv = list(GroupKFold(n_splits=n_splits).split(Xtr, y_train, groups))
    lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, penalty="l2",
                              scoring="roc_auc", max_iter=5000, n_jobs=-1)
    lr.fit(Xtr, y_train)
    return lr.predict_proba(Xte)[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    cols = [c for c in F_train.columns if not c.startswith(FAM_PREFIX)]   # RF: genera + markers + div
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _linear_scores(F_train, y_train, F_test, ctx)

    n = len(p_rf)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n
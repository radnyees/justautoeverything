"""Genus-level relative abundance + aggregate CRC oral-pathobiont marker features;
rank-averaged ensemble of a random forest (genera + markers) and an L2 logistic regression
on multi-resolution (genus + family) CLR-transformed abundances plus alpha-diversity summaries."""
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
DIV_PREFIX = "DIV_"    # alpha-diversity summaries, used by the linear model only
RARE_DEPTH = 1000      # depth for expected (analytic) rarefied richness


def _level(c, i, empty):
    parts = c.split(";")
    return parts[i] if len(parts) > i else empty


def _shannon(P):
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P), 0.0)
    return -t.sum(axis=1)


def _rarefied_richness(C, m):
    """Expected number of OTUs observed when subsampling m reads without replacement."""
    C = np.asarray(C, dtype=float)
    N = C.sum(axis=1)
    out = np.empty(len(C))
    for k in range(len(C)):
        n = C[k][C[k] > 0]
        Nk = N[k]
        if Nk <= m:
            out[k] = len(n)
            continue
        rest = Nk - n
        ok = rest >= m
        p_abs = np.zeros_like(n)
        p_abs[ok] = np.exp(gammaln(rest[ok] + 1) - gammaln(rest[ok] - m + 1)
                           - gammaln(Nk + 1) + gammaln(Nk - m + 1))
        out[k] = np.sum(1.0 - p_abs)
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

    P = rel.values
    D = pd.DataFrame(index=rel.index)
    D[DIV_PREFIX + "shannon_otu"] = _shannon(P)
    D[DIV_PREFIX + "log_invsimpson"] = np.log(1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None))
    gcols = [c for c in G.columns if not c.startswith("MK_")]
    D[DIV_PREFIX + "shannon_genus"] = _shannon(G[gcols].values)
    D[DIV_PREFIX + "log_rare_rich"] = np.log1p(_rarefied_richness(items.values, RARE_DEPTH))
    return pd.concat([G, Fm, D], axis=1)


def _clr(F, cols):
    """Centered log-ratio over the given columns (per sample, label-free)."""
    L = np.log(F.reindex(columns=cols, fill_value=0).values + PSEUDO)
    return L - L.mean(axis=1, keepdims=True)


def _linear_scores(F_train, y_train, F_test, ctx):
    extra = [c for c in F_train.columns if c.startswith("MK_") or c.startswith(DIV_PREFIX)]
    taxa = [c for c in F_train.columns if c not in set(extra)]   # genera + families
    prev = (F_train[taxa] > 0).mean(axis=0)
    keep = [g for g in taxa if prev[g] >= MIN_PREV]
    keep_g = [c for c in keep if not c.startswith(FAM_PREFIX)]
    keep_f = [c for c in keep if c.startswith(FAM_PREFIX)]
    Xtr = np.hstack([_clr(F_train, keep_g), _clr(F_train, keep_f), F_train[extra].values])
    Xte = np.hstack([_clr(F_test, keep_g), _clr(F_test, keep_f),
                     F_test.reindex(columns=extra, fill_value=0).values])
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
    cols = [c for c in F_train.columns
            if not c.startswith(FAM_PREFIX) and not c.startswith(DIV_PREFIX)]  # RF: genera + markers
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _linear_scores(F_train, y_train, F_test, ctx)

    n = len(p_rf)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n
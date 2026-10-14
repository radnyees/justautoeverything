"""Lineage-aware family+genus relative abundance + alpha diversity (depth-standardized richness,
Shannon, Simpson, evenness), all percentile-ranked within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def _lineage(col, depth):
    # Full lineage prefix up to `depth` levels (k,p,c,o,f = 5; +g = 6), so that
    # unannotated genera ('g__') stay distinct per family instead of being lumped together.
    parts = col.split(";")
    parts = parts + [""] * max(0, depth - len(parts))
    return ";".join(parts[:depth])


def _rarefied_richness(C, d_vec):
    # Expected number of taxa observed when subsampling d reads without replacement
    # (analytic rarefaction, Hurlbert 1971): sum_i 1 - C(N-n_i, d) / C(N, d).
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        n = C[i][C[i] > 0]
        N = n.sum()
        d = d_vec[i]
        if N <= d or len(n) == 0:
            out[i] = len(n)
            continue
        rest = N - n
        ok = rest >= d
        safe = np.where(ok, rest, d)
        logq = gammaln(safe + 1) + gammaln(N - d + 1) - gammaln(safe - d + 1) - gammaln(N + 1)
        out[i] = np.sum(np.where(ok, 1.0 - np.exp(logq), 1.0))
    return out


def _diversity(C):
    rel = C / np.clip(C.sum(axis=1, keepdims=True), 1, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(rel > 0, np.log(rel), 0.0)
    shannon = -(rel * lp).sum(axis=1)
    invsimp = 1.0 / np.clip((rel ** 2).sum(axis=1), 1e-12, None)
    obs = (C > 0).sum(axis=1)
    pielou = shannon / np.log(np.clip(obs, 2, None))
    return shannon, invsimp, pielou


def featurize(items, ctx):
    counts = items.fillna(0).clip(lower=0)
    rel = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
    blocks = []
    for depth in (5, 6):                                    # family, genus
        keys = np.array([_lineage(c, depth) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]

    studies = _study_labels(G, ctx)

    # Alpha diversity: IBD is characterized by reduced richness/evenness.
    C_otu = np.round(counts.values).astype(float)
    gkeys = np.array([_lineage(c, 6) for c in counts.columns])
    C_gen = np.round(counts.T.groupby(gkeys).sum().T.values).astype(float)
    depth_all = C_otu.sum(axis=1)
    d_vec = np.zeros(len(depth_all))
    for s in pd.unique(studies):
        m = studies == s
        # common rarefaction depth per study (10th percentile of depths, so few samples fall below)
        d_vec[m] = max(1.0, np.floor(np.percentile(depth_all[m], 10)))
    sh_o, is_o, pe_o = _diversity(C_otu)
    sh_g, is_g, _ = _diversity(C_gen)
    D = pd.DataFrame({
        "div_shannon_otu": sh_o, "div_invsimpson_otu": is_o, "div_pielou_otu": pe_o,
        "div_shannon_gen": sh_g, "div_invsimpson_gen": is_g,
        "div_rich_otu": _rarefied_richness(C_otu, d_vec),
        "div_rich_gen": _rarefied_richness(C_gen, d_vec),
    }, index=G.index)
    G = pd.concat([G, D], axis=1)

    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
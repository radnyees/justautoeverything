"""Lineage-aware phylum..genus robust-CLR, percentile-ranked within study + within-study dysbiosis
distances (median L1 distance on ranks, median Bray-Curtis distance on relative abundance, and
median Jaccard distance on per-study rarefied presence/absence, to training controls / cases of
the same study) + random forest."""
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier

RA_PREFIX = "RA|"
PA_PREFIX = "PA|"


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    return _study_from_meta(items, meta)


def _study_from_meta(items, meta):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object).astype(str)
    if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object).astype(str)


def _lineage(col, depth):
    # Full lineage prefix up to `depth` levels (k,p = 2; c = 3; o = 4; f = 5; g = 6), so that
    # unannotated levels (e.g. 'g__') stay distinct per parent taxon instead of being lumped together.
    parts = col.split(";")
    parts = parts + [""] * max(0, depth - len(parts))
    return ";".join(parts[:depth])


def _robust_clr(C):
    # rCLR: log(count) minus the mean log over the sample's non-zero taxa at this level.
    # No pseudocount (depth-independent); zeros become one tied minimum value.
    X = C.values.astype(float)
    nz = X > 0
    L = np.where(nz, np.log(np.where(nz, X, 1.0)), 0.0)
    n = nz.sum(axis=1)
    gm = np.where(n > 0, L.sum(axis=1) / np.maximum(n, 1), 0.0)
    R = np.where(nz, L - gm[:, None], -1e6)
    return pd.DataFrame(R, index=C.index, columns=C.columns)


def _rarefy_within_study(C, studies, q=0.10, seed=0):
    # Subsample each sample (without replacement) to its study's q-quantile depth so that
    # presence/absence is comparable within a study; shallower samples keep all reads.
    X = np.rint(C.values.astype(float)).clip(min=0).astype(np.int64)
    out = X.copy()
    rng = np.random.default_rng(seed)
    tot = X.sum(axis=1)
    for s in pd.unique(studies):
        m = np.where(studies == s)[0]
        target = int(max(np.quantile(tot[m], q), 1))
        for i in m:
            if tot[i] > target:
                out[i] = rng.multivariate_hypergeometric(X[i], target, method="marginals")
    return pd.DataFrame(out, index=C.index, columns=C.columns)


def featurize(items, ctx):
    blocks, ra_blocks = [], []
    studies = _study_labels(items, ctx)
    genus_counts = None
    for depth in (2, 3, 4, 5, 6):                           # phylum, class, order, family, genus
        keys = np.array([_lineage(c, depth) for c in items.columns])
        C = items.T.groupby(keys).sum().T
        blocks.append(_robust_clr(C))
        if depth in (5, 6):
            # Relative abundance (used only for Bray-Curtis dysbiosis distances, not as RF features).
            tot = C.sum(axis=1).values.astype(float)
            RA = C.values.astype(float) / np.where(tot > 0, tot, 1.0)[:, None]
            ra_blocks.append(pd.DataFrame(RA, index=C.index,
                                          columns=[f"{RA_PREFIX}{depth}|{c}" for c in C.columns]))
        if depth == 6:
            genus_counts = C
    # Presence/absence after within-study rarefaction (used only for Jaccard distances).
    R6 = _rarefy_within_study(genus_counts, studies)
    fam_keys = np.array([_lineage(c, 5) for c in R6.columns])
    R5 = R6.T.groupby(fam_keys).sum().T
    for d, R in ((6, R6), (5, R5)):
        ra_blocks.append(pd.DataFrame((R.values > 0).astype(np.float32), index=R.index,
                                      columns=[f"{PA_PREFIX}{d}|{c}" for c in R.columns]))
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    studies = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return pd.concat([out] + ra_blocks, axis=1)


def _median_rows(D):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(D, axis=1)
    if np.isnan(med).any():
        fill = np.nanmedian(med) if np.isfinite(med).any() else 0.0
        med = np.where(np.isnan(med), fill, med)
    return med


def _dist(A, B, metric):
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "jaccard":
            D = cdist(A > 0, B > 0, metric)
        else:
            D = cdist(A, B, metric)
    if metric == "cityblock":
        D = D / max(A.shape[1], 1)
    return D


def _dysbiosis_features(blocks, y, groups, s_tr, s_te):
    # For each (A_tr, A_te, metric) block: median distance to same-study training controls and
    # cases. Training rows exclude themselves and other samples of the same subject.
    classes = np.unique(y)
    pos = classes[-1]
    ftr, fte = [], []
    for A_tr, A_te, metric in blocks:
        if A_tr.shape[1] == 0:
            continue
        d_tr = {c: np.zeros(len(A_tr)) for c in (0, 1)}
        d_te = {c: np.zeros(len(A_te)) for c in (0, 1)}
        for s in pd.unique(np.concatenate([s_tr, s_te])):
            tr_i, te_i = s_tr == s, s_te == s
            for c, is_c in ((0, y != pos), (1, y == pos)):
                ref = tr_i & is_c
                if ref.sum() < 2:
                    ref = is_c
                if ref.sum() == 0:
                    continue
                if tr_i.any():
                    D = _dist(A_tr[tr_i], A_tr[ref], metric)
                    same = groups[tr_i][:, None] == groups[ref][None, :]
                    D[same] = np.nan
                    d_tr[c][tr_i] = _median_rows(D)
                if te_i.any():
                    D = _dist(A_te[te_i], A_tr[ref], metric)
                    d_te[c][te_i] = _median_rows(D)
        for a, b in ((d_tr[0], d_te[0]), (d_tr[1], d_te[1]), (d_tr[0] - d_tr[1], d_te[0] - d_te[1])):
            ftr.append(a.copy())
            fte.append(b.copy())
    Ftr, Fte = np.column_stack(ftr), np.column_stack(fte)
    Ftr = np.nan_to_num(Ftr)
    Fte = np.nan_to_num(Fte)
    # Per-study standardization using training rows only (label-free statistics).
    for s in pd.unique(s_tr):
        tr_i, te_i = s_tr == s, s_te == s
        mu = Ftr[tr_i].mean(axis=0)
        sd = Ftr[tr_i].std(axis=0) + 1e-9
        Ftr[tr_i] = (Ftr[tr_i] - mu) / sd
        if te_i.any():
            Fte[te_i] = (Fte[te_i] - mu) / sd
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    all_cols = F_train.columns
    F_te = F_test.reindex(columns=all_cols, fill_value=0)
    is_aux = np.array([str(c).startswith(RA_PREFIX) or str(c).startswith(PA_PREFIX)
                       for c in all_cols])
    cols = all_cols[~is_aux]                                # ranked rCLR taxa (RF features)
    Xtr, Xte = F_train[cols].values, F_te[cols].values
    y = np.asarray(y_train)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else np.arange(len(y))
    s_tr = _study_from_meta(F_train, ctx.get("meta_train"))
    s_te = _study_from_meta(F_test, ctx.get("meta_test"))

    depth = np.array([c.count(";") + 1 for c in cols])
    blocks = [(Xtr[:, m], Xte[:, m], "cityblock")
              for m in (depth == 6, depth == 5, np.ones(len(cols), dtype=bool))]
    aux_cols = all_cols[is_aux]
    for prefix, metric in ((RA_PREFIX, "braycurtis"), (PA_PREFIX, "jaccard")):
        for d in (6, 5):
            rc = [c for c in aux_cols if c.startswith(f"{prefix}{d}|")]
            if rc:
                blocks.append((F_train[rc].values.astype(float), F_te[rc].values.astype(float),
                               metric))

    Dtr, Dte = _dysbiosis_features(blocks, y, groups, s_tr, s_te)
    Xtr = np.hstack([Xtr, Dtr])
    Xte = np.hstack([Xte, Dte])
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    return rf.predict_proba(Xte)[:, 1]
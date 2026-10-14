"""Lineage-aware phylum..genus robust-CLR, percentile-ranked within study + within-study dysbiosis
distances (median and k-nearest-neighbour L1 distance to training controls / cases of the same
study) + random forest."""
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier


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


def featurize(items, ctx):
    blocks = []
    for depth in (2, 3, 4, 5, 6):                           # phylum, class, order, family, genus
        keys = np.array([_lineage(c, depth) for c in items.columns])
        C = items.T.groupby(keys).sum().T
        blocks.append(_robust_clr(C))
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    studies = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def _median_rows(D):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(D, axis=1)
    if np.isnan(med).any():
        fill = np.nanmedian(med) if np.isfinite(med).any() else 0.0
        med = np.where(np.isnan(med), fill, med)
    return med


def _knn_rows(D, k):
    # Mean of the k smallest non-NaN distances per row (local nearest-neighbour distance).
    S = np.sort(np.where(np.isnan(D), np.inf, D), axis=1)[:, :k]
    fin = np.isfinite(S)
    cnt = fin.sum(axis=1)
    out = np.where(cnt > 0, np.where(fin, S, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    if np.isnan(out).any():
        fill = np.nanmedian(out) if np.isfinite(out).any() else 0.0
        out = np.where(np.isnan(out), fill, out)
    return out


def _dysbiosis_features(Xtr, Xte, cols, y, groups, s_tr, s_te):
    # Median and kNN L1 distance (on within-study ranks) to same-study training controls and cases.
    # Training rows exclude themselves and other samples of the same subject.
    depth = np.array([c.count(";") + 1 for c in cols])
    masks = [depth == 6, depth == 5, np.ones(len(cols), dtype=bool)]
    classes = np.unique(y)
    pos = classes[-1]
    ftr, fte = [], []
    for m in masks:
        if m.sum() == 0:
            continue
        A_tr, A_te = Xtr[:, m], Xte[:, m]
        d_tr = {c: np.zeros(len(Xtr)) for c in (0, 1)}
        d_te = {c: np.zeros(len(Xte)) for c in (0, 1)}
        k_tr = {c: np.zeros(len(Xtr)) for c in (0, 1)}
        k_te = {c: np.zeros(len(Xte)) for c in (0, 1)}
        for s in pd.unique(np.concatenate([s_tr, s_te])):
            tr_i, te_i = s_tr == s, s_te == s
            for c, is_c in ((0, y != pos), (1, y == pos)):
                ref = tr_i & is_c
                if ref.sum() < 2:
                    ref = is_c
                if ref.sum() == 0:
                    continue
                k = int(np.clip(round(np.sqrt(ref.sum())), 3, 15))
                k = max(1, min(k, int(ref.sum())))
                if tr_i.any():
                    D = cdist(A_tr[tr_i], A_tr[ref], "cityblock") / m.sum()
                    same = groups[tr_i][:, None] == groups[ref][None, :]
                    D[same] = np.nan
                    d_tr[c][tr_i] = _median_rows(D)
                    k_tr[c][tr_i] = _knn_rows(D, k)
                if te_i.any():
                    D = cdist(A_te[te_i], A_tr[ref], "cityblock") / m.sum()
                    d_te[c][te_i] = _median_rows(D)
                    k_te[c][te_i] = _knn_rows(D, k)
        for a, b in ((d_tr[0], d_te[0]), (d_tr[1], d_te[1]), (d_tr[0] - d_tr[1], d_te[0] - d_te[1]),
                     (k_tr[0], k_te[0]), (k_tr[1], k_te[1]), (k_tr[0] - k_tr[1], k_te[0] - k_te[1])):
            ftr.append(a.copy())
            fte.append(b.copy())
    Ftr, Fte = np.column_stack(ftr), np.column_stack(fte)
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
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else np.arange(len(y))
    s_tr = _study_from_meta(F_train, ctx.get("meta_train"))
    s_te = _study_from_meta(F_test, ctx.get("meta_test"))
    Dtr, Dte = _dysbiosis_features(Xtr, Xte, list(cols), y, groups, s_tr, s_te)
    Xtr = np.hstack([Xtr, Dtr])
    Xte = np.hstack([Xte, Dte])
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    return rf.predict_proba(Xte)[:, 1]
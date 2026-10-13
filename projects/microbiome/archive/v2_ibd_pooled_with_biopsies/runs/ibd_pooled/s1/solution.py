"""Genus-level relative abundance; rank-average of (a) random forest on proportions
(min_samples_leaf=5 so leaves cannot isolate single heavily-weighted samples) and
(b) L2 logistic regression on CLR-transformed prevalent genera plus the Gevers 2014 microbial
dysbiosis index (log enriched/depleted families) and the HMP2 (Lloyd-Price 2019) dysbiosis
score (median Bray-Curtis distance to same-study training controls, own subject excluded),
harmonized WITHIN each study by a rank-based inverse normal transform (label-free, robust to
zero-inflation and study-specific tails from region/depth/protocol). Both use (study, class)-
balanced sample weights so the pooled models cannot gain from study-specific case/control ratios."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MDI_COL = "__MDI__"
# Gevers et al. 2014 (RISK) microbial dysbiosis index components
_MDI_UP_FAMILIES = ("enterobacteriaceae", "pasteurellaceae", "fusobacteriaceae",
                    "neisseriaceae", "veillonellaceae", "gemellaceae")
_MDI_DOWN_FAMILIES = ("erysipelotrichaceae", "bifidobacteriaceae")
_MDI_DOWN_ORDERS = ("bacteroidales", "clostridiales")


def _level(parts, i):
    if len(parts) <= i:
        return ""
    return parts[i].split("__", 1)[-1].strip("[] ").lower()


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    parts = [c.split(";") for c in rel.columns]
    genus = [p[5] if len(p) > 5 else "g__" for p in parts]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    up = np.zeros(len(parts), dtype=bool)
    down = np.zeros(len(parts), dtype=bool)
    for j, p in enumerate(parts):
        o, f = _level(p, 3), _level(p, 4)
        if f in _MDI_UP_FAMILIES:
            up[j] = True
        elif f in _MDI_DOWN_FAMILIES or o in _MDI_DOWN_ORDERS:
            down[j] = True
    R = rel.values
    eps = 1e-5
    mdi = np.log(R[:, up].sum(axis=1) + eps) - np.log(R[:, down].sum(axis=1) + eps)
    G = G.copy()
    G[MDI_COL] = mdi
    return G


def _balanced_weights(y, study):
    """Weight each sample by 1 / size of its (study, class) cell; normalized to mean 1."""
    y = np.asarray(y)
    study = np.asarray(study).astype(str)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(study):
        for c in np.unique(y):
            m = (study == s) & (y == c)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w * (len(w) / w.sum())


def _clr(X, eps=1e-5):
    L = np.log(np.asarray(X, dtype=float) + eps)
    return L - L.mean(axis=1, keepdims=True)


def _study_rank_normal(Z, study, min_n=5):
    """Rank-based inverse normal transform of each feature within each study (ties -> mid-rank);
    tiny studies fall back to global z-scoring."""
    Z = np.array(Z, dtype=float, copy=True)
    out = np.empty_like(Z)
    g_mu, g_sd = Z.mean(axis=0), Z.std(axis=0)
    g_sd[g_sd < 1e-8] = 1.0
    for s in np.unique(study):
        m = study == s
        n = int(m.sum())
        if n >= min_n:
            r = rankdata(Z[m], axis=0)                       # average ranks per column
            out[m] = norm.ppf((r - 0.5) / n)
        else:
            out[m] = (Z[m] - g_mu) / g_sd
    return out


def _get_study(ctx, key, n):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
    if meta is not None and "study" in meta and len(meta) == n:
        return np.asarray(meta["study"].values).astype(str)
    return np.zeros(n).astype(str)


def _dysbiosis_score(Xtr, Xte, y, study_tr, study_te, groups, min_ctrl=3):
    """HMP2-style dysbiosis: median Bray-Curtis distance of each sample to the training controls
    of its own study (controls from the same subject excluded)."""
    ctrl_all = y == np.min(y)
    d_tr = np.full(len(Xtr), np.nan)
    d_te = np.full(len(Xte), np.nan)
    for s in np.unique(np.concatenate([study_tr, study_te])):
        ctrl = ctrl_all & (study_tr == s)
        if ctrl.sum() < min_ctrl:
            ctrl = ctrl_all
        ref = Xtr[ctrl]
        ref_g = groups[ctrl]
        mtr = np.where(study_tr == s)[0]
        if len(mtr):
            D = cdist(Xtr[mtr], ref, metric="braycurtis")
            D[groups[mtr][:, None] == ref_g[None, :]] = np.nan
            ok = ~np.all(np.isnan(D), axis=1)
            d_tr[mtr[ok]] = np.nanmedian(D[ok], axis=1)
        mte = np.where(study_te == s)[0]
        if len(mte):
            D = cdist(Xte[mte], ref, metric="braycurtis")
            d_te[mte] = np.median(D, axis=1)
    fill = np.nanmedian(d_tr) if np.any(~np.isnan(d_tr)) else 0.5
    d_tr[np.isnan(d_tr)] = fill
    d_te[np.isnan(d_te)] = fill
    return d_tr[:, None], d_te[:, None]


def fit_predict(F_train, y_train, F_test, ctx):
    cols = [c for c in F_train.columns if c != MDI_COL]     # align test genera to training genera
    Xtr = np.asarray(F_train[cols].values, dtype=float)
    Xte = np.asarray(F_test.reindex(columns=cols, fill_value=0).values, dtype=float)
    mdi_tr = np.asarray(F_train[MDI_COL].values, dtype=float)[:, None]
    mdi_te = np.asarray(F_test[MDI_COL].values, dtype=float)[:, None]
    y = np.asarray(y_train)

    study_tr = _get_study(ctx, "meta_train", len(y))
    study_te = _get_study(ctx, "meta_test", len(Xte))
    w = _balanced_weights(y, study_tr)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    if groups is None or len(groups) != len(y):
        groups = np.arange(len(y))
    groups = np.asarray(groups).astype(str)

    # (a) random forest on genus proportions; leaves must hold >=5 samples so heavily
    # weighted samples from small studies cannot each get their own leaf
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # (b) L2 logistic regression on per-study rank-normalized CLR of genera present in >=5% of
    # training, plus the per-study rank-normalized dysbiosis index and HMP2 dysbiosis score
    keep = (Xtr > 0).mean(axis=0) >= 0.05
    if keep.sum() < 2:
        keep = np.ones(Xtr.shape[1], dtype=bool)
    dys_tr, dys_te = _dysbiosis_score(Xtr, Xte, y, study_tr, study_te, groups)
    Ztr = np.hstack([_clr(Xtr[:, keep]), mdi_tr, dys_tr])
    Zte = np.hstack([_clr(Xte[:, keep]), mdi_te, dys_te])
    Ztr = _study_rank_normal(Ztr, study_tr)
    Zte = _study_rank_normal(Zte, study_te)
    lr = LogisticRegression(C=0.05, penalty="l2", max_iter=5000)
    lr.fit(Ztr, y, sample_weight=w)
    p_lr = lr.decision_function(Zte)

    n = len(p_rf)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n
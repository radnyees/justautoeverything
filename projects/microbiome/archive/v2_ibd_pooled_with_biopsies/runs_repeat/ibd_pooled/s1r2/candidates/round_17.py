"""Genus-level relative abundance, per-study percentile-rank harmonization with zeros mapped to 0
(so 'not detected' is identical across studies), plus label-free co-abundance-group (CAG)
features: prevalent genera clustered on study-averaged within-study Spearman correlation, each
CAG scored as the mean member rank and re-ranked within study. In fit_predict a reference-based
dysbiosis score (Lloyd-Price 2019 style) is added: median Bray-Curtis distance (genus relative
abundance) to same-study training controls and to training cases, leave-own-subject-out, mapped
through the study's training ECDF. Ensemble of a random forest, an L2 logistic regression (inner
grouped CV for C) and a shallow regularized LightGBM, all trained with per-(study, class)
balanced sample weights so study prevalence is not learnable. Predictions are rank-averaged."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

RA_PREFIX = "RA__"


def _study_from_meta(meta, index, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    try:
        s = meta["study"].reindex(index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == n:
            return meta["study"].astype(str).values
        return np.zeros(n, dtype=object)


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    return _study_from_meta(meta, items.index, len(items))


def _cag_features(vals, Gv, studies, n_clusters=20, min_prev=0.10):
    """Co-abundance groups: genera prevalent (>=min_prev) in at least one study are clustered on
    the size-weighted mean of their within-study Spearman correlations (computed on the
    zero-mapped ranks); each group's score is the mean member rank, re-ranked within study."""
    uniq = pd.unique(studies)
    prev_ok = np.zeros(Gv.shape[1], dtype=bool)
    for s in uniq:
        m = np.asarray(studies == s)
        prev_ok |= (Gv[m] > 0).mean(axis=0) >= min_prev
    idx = np.where(prev_ok)[0]
    if len(idx) < n_clusters + 2:
        return None
    C = np.zeros((len(idx), len(idx)))
    wsum = 0.0
    for s in uniq:
        m = np.asarray(studies == s)
        if m.sum() < 10:
            continue
        sub = vals[m][:, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(sub, rowvar=False)
        r = np.nan_to_num(r, nan=0.0)
        C += m.sum() * r
        wsum += m.sum()
    if wsum == 0:
        return None
    C /= wsum
    D = np.clip(1.0 - C, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=n_clusters, criterion="maxclust")
    raw = np.column_stack([vals[:, idx[lab == k]].mean(axis=1) for k in np.unique(lab)])
    out = np.zeros_like(raw)
    for s in uniq:
        m = np.asarray(studies == s)
        out[m] = np.array(pd.DataFrame(raw[m]).rank(axis=0, method="average").values,
                          dtype=float, copy=True) / max(m.sum(), 1)
    return out


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    vals = np.zeros(G.shape, dtype=float)
    Gv = np.array(G.values, dtype=float, copy=True)
    for s in pd.unique(studies):
        m = np.asarray(studies == s)
        sub = Gv[m]
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        r = np.array(pd.DataFrame(sub).rank(axis=0, method="average").values,
                     dtype=float, copy=True) / max(m.sum(), 1)
        r[sub <= 0] = 0.0                                   # undetected -> 0 in every study
        vals[m] = r
    F = pd.DataFrame(vals, index=G.index, columns=[str(c) for c in G.columns])
    cag = _cag_features(vals, Gv, studies)
    if cag is not None:
        Fc = pd.DataFrame(cag, index=G.index,
                          columns=["CAG_%d" % k for k in range(cag.shape[1])])
        F = pd.concat([F, Fc], axis=1)
    # raw genus relative abundances, used only for within-study Bray-Curtis distances
    Fr = pd.DataFrame(Gv, index=G.index, columns=[RA_PREFIX + str(c) for c in G.columns])
    return pd.concat([F, Fr], axis=1)


def _balanced_weights(studies, y):
    """Weight 1/(n in study x class cell), rescaled to mean 1: each study contributes
    equal case and control mass, removing study-prevalence as a learnable shortcut."""
    y = np.asarray(y)
    df = pd.DataFrame({"s": studies, "y": y})
    cell = df.groupby(["s", "y"])["y"].transform("size").values.astype(float)
    n_cls = df.groupby("s")["y"].transform("nunique").values.astype(float)
    study_n = df.groupby("s")["y"].transform("size").values.astype(float)
    # each study keeps total mass proportional to its size, split equally across its classes
    w = study_n / (n_cls * cell)
    return w * (len(w) / w.sum())


def _within_study_auc(y, p, studies):
    aucs = []
    for s in pd.unique(studies):
        m = studies == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return np.mean(aucs) if aucs else 0.5


def _ecdf(ref, x):
    sv = np.sort(ref)
    return (np.searchsorted(sv, x, "left") + np.searchsorted(sv, x, "right")) / (2.0 * len(sv))


def _dysbiosis_features(Rtr, y, st_tr, g_tr, Rte, st_te):
    """Median Bray-Curtis distance to same-study training controls / cases (own subject excluded
    for training samples), plus their difference; each mapped through its study's training ECDF."""
    with np.errstate(invalid="ignore", divide="ignore"):
        Dtr = np.nan_to_num(cdist(Rtr, Rtr, "braycurtis"), nan=1.0)
        Dte = np.nan_to_num(cdist(Rte, Rtr, "braycurtis"), nan=1.0)

    def ref_mask(s, c):
        m = (st_tr == s) & (y == c)
        return m if m.sum() >= 3 else (y == c)

    ftr = np.zeros((len(Rtr), 2))
    fte = np.zeros((len(Rte), 2))
    for c in (0, 1):
        if (y == c).sum() == 0:
            continue
        cache = {}
        for i in range(len(Rtr)):
            s = st_tr[i]
            if s not in cache:
                cache[s] = ref_mask(s, c)
            m = cache[s] & (g_tr != g_tr[i])
            if m.sum() == 0:
                m = (y == c) & (g_tr != g_tr[i])
            ftr[i, c] = np.median(Dtr[i, m]) if m.sum() > 0 else 1.0
        for i in range(len(Rte)):
            s = st_te[i]
            if s not in cache:
                cache[s] = ref_mask(s, c)
            m = cache[s]
            fte[i, c] = np.median(Dte[i, m])
    ftr = np.column_stack([ftr, ftr[:, 0] - ftr[:, 1]])
    fte = np.column_stack([fte, fte[:, 0] - fte[:, 1]])
    otr, ote = np.zeros_like(ftr), np.zeros_like(fte)
    for s in pd.unique(np.concatenate([st_tr, st_te])):
        mtr, mte = st_tr == s, st_te == s
        ref = ftr[mtr] if mtr.sum() >= 5 else ftr
        for j in range(ftr.shape[1]):
            if mtr.any():
                otr[mtr, j] = _ecdf(ref[:, j], ftr[mtr, j])
            if mte.any():
                ote[mte, j] = _ecdf(ref[:, j], fte[mte, j])
    return otr, ote


def _fit_lr(Xtr, y, w, studies, groups, Xte):
    """L2 logistic regression; C chosen by grouped inner CV on within-study AUROC."""
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1]
    if groups is not None and len(np.unique(groups)) >= 4:
        splits = list(GroupKFold(n_splits=4).split(Xtr, y, groups))
    else:
        splits = list(StratifiedKFold(4, shuffle=True, random_state=0).split(Xtr, y))
    scores = np.zeros(len(Cs))
    for tr, va in splits:
        sc = StandardScaler().fit(Xtr[tr])
        A, B = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
        wi = _balanced_weights(studies[tr], y[tr])
        for k, C in enumerate(Cs):
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(A, y[tr], sample_weight=wi)
            scores[k] += _within_study_auc(y[va], lr.decision_function(B), studies[va])
    C = Cs[int(np.argmax(scores))]
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit(sc.transform(Xtr), y, sample_weight=w)
    return lr.decision_function(sc.transform(Xte))


def _fit_lgb(Xtr, y, w, Xte):
    """Shallow, heavily subsampled boosting: captures conditional/threshold interactions."""
    gbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
        reg_lambda=5.0, random_state=0, n_jobs=4, deterministic=True,
        force_col_wise=True, verbose=-1)
    gbm.fit(Xtr, y, sample_weight=w)
    return gbm.predict_proba(Xte)[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    all_cols = list(F_train.columns)
    ra_cols = [c for c in all_cols if str(c).startswith(RA_PREFIX)]
    cols = [c for c in all_cols if not str(c).startswith(RA_PREFIX)]
    Xtr = np.array(F_train[cols].values, dtype=float, copy=True)
    Xte = np.array(F_test.reindex(columns=cols, fill_value=0).values, dtype=float, copy=True)
    y = np.asarray(y_train).astype(int)
    meta_tr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    meta_te = ctx.get("meta_test") if isinstance(ctx, dict) else None
    studies_tr = _study_from_meta(meta_tr, F_train.index, len(F_train))
    studies_te = _study_from_meta(meta_te, F_test.index, len(F_test))
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else None
    w = _balanced_weights(studies_tr, y)

    if ra_cols:
        Rtr = np.array(F_train[ra_cols].values, dtype=float, copy=True)
        Rte = np.array(F_test.reindex(columns=ra_cols, fill_value=0).values, dtype=float,
                       copy=True)
        g_tr = groups if groups is not None else np.arange(len(y))
        dtr, dte = _dysbiosis_features(Rtr, y, np.asarray(studies_tr), g_tr,
                                       Rte, np.asarray(studies_te))
        Xtr = np.hstack([Xtr, dtr])
        Xte = np.hstack([Xte, dte])

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _fit_lr(Xtr, y, w, studies_tr, groups, Xte)

    p_gb = _fit_lgb(Xtr, y, w, Xte)

    n = len(Xte)
    return (rankdata(p_rf) + rankdata(p_lr) + rankdata(p_gb)) / (3.0 * n)
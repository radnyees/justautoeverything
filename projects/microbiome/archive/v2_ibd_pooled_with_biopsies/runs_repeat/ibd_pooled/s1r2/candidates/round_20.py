"""Genus-level relative abundance, per-study percentile-rank harmonization with zeros mapped to 0
(so 'not detected' is identical across studies), plus label-free co-abundance-group (CAG)
features at two resolutions (coarse guilds and finer groups cut from the same dendrogram):
prevalent genera clustered on study-averaged within-study Spearman correlation, each CAG scored
as the mean member rank and re-ranked within study. Ensemble of a random forest, an L2 logistic
regression (inner grouped CV for C) and a shallow regularized LightGBM, all trained with
per-(study, class) balanced sample weights whose per-study mass is tempered to sqrt(study size),
matching the equal-per-study within-study AUROC metric. Predictions are rank-averaged."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


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


def _cag_features(vals, Gv, studies, resolutions=(6, 20), min_prev=0.10):
    """Co-abundance groups: genera prevalent (>=min_prev) in at least one study are clustered on
    the size-weighted mean of their within-study Spearman correlations (computed on the
    zero-mapped ranks); the dendrogram is cut at several resolutions (coarse guilds + finer
    groups); each group's score is the mean member rank, re-ranked within study."""
    uniq = pd.unique(studies)
    prev_ok = np.zeros(Gv.shape[1], dtype=bool)
    for s in uniq:
        m = np.asarray(studies == s)
        prev_ok |= (Gv[m] > 0).mean(axis=0) >= min_prev
    idx = np.where(prev_ok)[0]
    if len(idx) < max(resolutions) + 2:
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
    blocks, names = [], []
    for k_res in resolutions:
        lab = fcluster(Z, t=k_res, criterion="maxclust")
        raw = np.column_stack([vals[:, idx[lab == k]].mean(axis=1) for k in np.unique(lab)])
        out = np.zeros_like(raw)
        for s in uniq:
            m = np.asarray(studies == s)
            out[m] = np.array(pd.DataFrame(raw[m]).rank(axis=0, method="average").values,
                              dtype=float, copy=True) / max(m.sum(), 1)
        blocks.append(out)
        names += ["CAG%d_%d" % (k_res, j) for j in range(out.shape[1])]
    return np.hstack(blocks), names


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
        arr, names = cag
        Fc = pd.DataFrame(arr, index=G.index, columns=names)
        F = pd.concat([F, Fc], axis=1)
    return F


def _balanced_weights(studies, y, study_power=0.5):
    """Weight study_n**power / (n_classes x n in study x class cell), rescaled to mean 1: each
    study contributes equal case and control mass (study prevalence is not learnable), and its
    total mass grows only as sqrt(size), since the metric averages AUROC equally over studies."""
    y = np.asarray(y)
    df = pd.DataFrame({"s": studies, "y": y})
    cell = df.groupby(["s", "y"])["y"].transform("size").values.astype(float)
    n_cls = df.groupby("s")["y"].transform("nunique").values.astype(float)
    study_n = df.groupby("s")["y"].transform("size").values.astype(float)
    w = study_n ** study_power / (n_cls * cell)
    return w * (len(w) / w.sum())


def _within_study_auc(y, p, studies):
    aucs = []
    for s in pd.unique(studies):
        m = studies == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return np.mean(aucs) if aucs else 0.5


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
    cols = F_train.columns                                  # align test features to training ones
    Xtr = np.array(F_train[cols].values, dtype=float, copy=True)
    Xte = np.array(F_test.reindex(columns=cols, fill_value=0).values, dtype=float, copy=True)
    y = np.asarray(y_train).astype(int)
    meta_tr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    studies_tr = _study_from_meta(meta_tr, F_train.index, len(F_train))
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else None
    w = _balanced_weights(studies_tr, y)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _fit_lr(Xtr, y, w, studies_tr, groups, Xte)

    p_gb = _fit_lgb(Xtr, y, w, Xte)

    n = len(Xte)
    return (rankdata(p_rf) + rankdata(p_lr) + rankdata(p_gb)) / (3.0 * n)
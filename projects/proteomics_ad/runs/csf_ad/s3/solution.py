"""log2 + per-sample median centring + removal of a label-free blood-contamination axis
(hemoglobin / carbonic anhydrase 1 score regressed out of every protein) + half-minimum
imputation + per-protein rank-based inverse-normal transform (fitted on the training fold,
test samples mapped through the interpolated training ECDF), re-standardized and winsorized
at +-3 SD + L2 logistic regression with a biology prior (known AD CSF markers get a weaker L2
penalty and a relaxed completeness filter), blended with a low-variance diagonal-LDA score
(per-protein standardized mean differences, same prior boost) that ignores protein
correlations. Prior strength, C and the blend weight are chosen per training fold by grouped
inner CV scored with within-site AUROC."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# Proteins repeatedly reported as altered in AD CSF (tau/glycolysis/14-3-3 module,
# neuronal injury & synapse, glial activation, complement, lipid transport, APP/tau).
PRIOR_GENES = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "YWHAH",
    "PKM", "ALDOA", "ALDOC", "GAPDH", "ENO1", "ENO2", "MDH1", "LDHB", "PGK1", "TPI1",
    "PEBP1", "SOD1", "CHGA", "SCG2",
    "NEFL", "NEFM", "NRGN", "SNAP25", "NPTX2", "NPTXR", "VGF", "MAPT", "APP",
    "CHI3L1", "CHIT1", "SPP1", "GFAP", "TREM2",
    "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH",
    "APOE", "CLU",
    # extended tau-correlated metabolic / cytoskeletal module (Bader 2020, Higginbotham 2020)
    "GOT1", "LDHA", "PGAM1", "PPIA", "PARK7", "FABP3", "TMSB10", "TMSB4X", "GDA", "CD44",
}
# Erythrocyte-specific proteins marking blood contamination of CSF (Geyer 2019, Bader 2020).
BLOOD_GENES = {"HBA1", "HBA2", "HBB", "HBD", "CA1"}
MIN_PRESENT = 0.7          # completeness required for generic proteins
MIN_PRESENT_PRIOR = 0.3    # relaxed completeness for prior-biology proteins
CLIP_Z = 3.0               # winsorize standardized features at +-3 SD
SCALE_GRID = (1.5, 2.0, 3.0)
C_GRID = (0.03, 0.05, 0.1)
ALPHA_GRID = (0.0, 0.25, 0.5)   # weight of the diagonal-LDA score in the blend
DEFAULT = (2.0, 0.05)
DEFAULT_ALPHA = 0.25


def _genes(col):
    return {g.strip().upper() for g in str(col).split("|")[0].split(";")}


def _is_prior(col):
    return bool(_genes(col) & PRIOR_GENES)


def _remove_blood_axis(Lc):
    bcols = [c for c in Lc.columns if _genes(c) & BLOOD_GENES]
    if not bcols:
        return Lc
    B = Lc[bcols]
    Z = (B - B.mean()) / B.std().replace(0, np.nan)
    s = Z.mean(axis=1, skipna=True)
    s = s.fillna(s.median()).fillna(0.0).values
    s = s - s.mean()
    X = Lc.values.astype(float)
    M = ~np.isnan(X)
    n = M.sum(0)
    S = np.broadcast_to(s[:, None], X.shape)
    sm = np.where(n > 0, (M * S).sum(0) / np.maximum(n, 1), 0.0)
    sx = (S - sm) * M
    xm = np.where(n > 0, np.nansum(X, 0) / np.maximum(n, 1), 0.0)
    xc = np.where(M, X - xm, 0.0)
    den = (sx ** 2).sum(0)
    slope = np.where((n >= 10) & (den > 1e-12), (sx * xc).sum(0) / np.maximum(den, 1e-12), 0.0)
    Xr = X - slope * (S - sm)                               # NaNs stay NaN
    return pd.DataFrame(Xr, index=Lc.index, columns=Lc.columns)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    Lc = L.sub(L.median(axis=1), axis=0)                   # per-sample median centring
    return _remove_blood_axis(Lc)


def _rank_int(Xtr, Xte):
    """Per-protein rank-based inverse-normal transform fitted on the training samples.
    Ties (e.g. the imputed floor) get mid-ranks; test values are mapped through the
    linearly interpolated training ECDF (clamped at the training extremes)."""
    n = Xtr.shape[0]
    Utr = np.empty_like(Xtr, dtype=float)
    Ute = np.empty_like(Xte, dtype=float)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        xs = np.sort(col)
        Utr[:, j] = ((np.searchsorted(xs, col, "left") + np.searchsorted(xs, col, "right"))
                     / 2.0 + 0.5) / (n + 1)
        ux = np.unique(xs)
        uu = ((np.searchsorted(xs, ux, "left") + np.searchsorted(xs, ux, "right"))
              / 2.0 + 0.5) / (n + 1)
        Ute[:, j] = np.interp(Xte[:, j], ux, uu)
    Ztr, Zte = norm.ppf(Utr), norm.ppf(Ute)
    mu = Ztr.mean(0)
    sd = Ztr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Ztr - mu) / sd, (Zte - mu) / sd


def _prep(F_train, F_test):
    present = F_train.notna().mean()
    prior = np.array([_is_prior(c) for c in F_train.columns])
    thr = np.where(prior, MIN_PRESENT_PRIOR, MIN_PRESENT)
    keep = F_train.columns[present.values >= thr]
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)
    pmask = np.array([_is_prior(c) for c in keep])
    Ztr, Zte = _rank_int(Xtr, Xte)
    Ztr = np.clip(Ztr, -CLIP_Z, CLIP_Z)
    Zte = np.clip(Zte, -CLIP_Z, CLIP_Z)
    return Ztr, Zte, pmask


def _zscore_by_train(s_tr, s_te):
    sd = s_tr.std()
    return (s_te - s_tr.mean()) / (sd if sd > 1e-12 else 1.0)


def _fit_pred(Ztr, y, Zte, pmask, scale, C):
    """Returns train-standardized logistic and diagonal-LDA scores for the test samples."""
    w = np.where(pmask, scale, 1.0)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Ztr * w, y)
    lr = _zscore_by_train(clf.decision_function(Ztr * w), clf.decision_function(Zte * w))
    # diagonal LDA: standardized mean difference per protein, prior boosted like the penalty
    a, b = Ztr[y == 1], Ztr[y == 0]
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1))
                     / max(na + nb - 2, 1))
    d = (a.mean(0) - b.mean(0)) / np.maximum(pooled, 1e-6)
    coef = d * w ** 2
    dl = _zscore_by_train(Ztr @ coef, Zte @ coef)
    return lr, dl


def _blend(scores, alpha):
    lr, dl = scores
    return (1.0 - alpha) * lr + alpha * dl


def _sites(ctx, n):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    try:
        if meta is not None and "site" in meta:
            s = np.asarray(meta["site"])
            if len(s) == n:
                return s
    except Exception:
        pass
    return np.zeros(n)


def _within_site_auc(y, p, sites):
    tot, wsum = 0.0, 0.0
    for s in np.unique(sites):
        m = sites == s
        if len(np.unique(y[m])) < 2:
            continue
        tot += m.sum() * roc_auc_score(y[m], p[m])
        wsum += m.sum()
    return tot / wsum if wsum > 0 else np.nan


def _select(F_train, y, ctx):
    n = len(y)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == n else np.arange(n)
    sites = _sites(ctx, n)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 3:
        return DEFAULT, DEFAULT_ALPHA
    grid = [(s, c) for s in SCALE_GRID for c in C_GRID]
    oof_lr = {g: np.full(n, np.nan) for g in grid}
    oof_dl = {g: np.full(n, np.nan) for g in grid}
    for tr, va in GroupKFold(n_splits=n_splits).split(np.zeros(n), y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        Ztr, Zva, pmask = _prep(F_train.iloc[tr], F_train.iloc[va])
        for g in grid:
            lr, dl = _fit_pred(Ztr, y[tr], Zva, pmask, *g)
            oof_lr[g][va] = lr
            oof_dl[g][va] = dl
    # step 1: choose (prior scale, C) for the logistic model
    best, best_score = DEFAULT, -np.inf
    for g in grid:
        p = oof_lr[g]
        if np.isnan(p).any():
            continue
        sc = _within_site_auc(y, p, sites)
        if np.isfinite(sc) and sc > best_score + 1e-9:
            best, best_score = g, sc
    if not np.isfinite(best_score):
        return best, DEFAULT_ALPHA
    # step 2: choose the diagonal-LDA blend weight given the selected configuration
    best_alpha, best_a_score = 0.0, -np.inf
    for alpha in ALPHA_GRID:
        p = _blend((oof_lr[best], oof_dl[best]), alpha)
        sc = _within_site_auc(y, p, sites)
        if np.isfinite(sc) and sc > best_a_score + 1e-9:
            best_alpha, best_a_score = alpha, sc
    return best, best_alpha


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    (scale, C), alpha = _select(F_train, y, ctx)
    Ztr, Zte, pmask = _prep(F_train, F_test)
    return _blend(_fit_pred(Ztr, y, Zte, pmask, scale, C), alpha)
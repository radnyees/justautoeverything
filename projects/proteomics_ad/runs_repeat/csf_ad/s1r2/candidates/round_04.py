"""log2 + per-sample median centring + half-minimum imputation + L2 logistic regression,
rank-blended with a prior-biology two-sided CSF module score:
mean z of 'tau-associated, up-in-AD' proteins minus mean z of 'synaptic, down-in-AD' proteins.
The blend weight is chosen by grouped inner CV (within-site AUROC) on the training fold."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Proteins repeatedly reported as increased in AD CSF (tau/neuronal injury, glycolysis, glia).
AD_UP_PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "PKM", "ALDOA", "GAPDH", "ENO1", "ENO2", "MDH1",
    "LDHB", "PGK1", "TPI1", "PPIA", "CHI3L1", "SPP1", "NEFL", "MAPT", "GFAP",
}

# Synaptic / neurosecretory proteins repeatedly reported as decreased in clinical AD CSF.
AD_DOWN_PANEL = {"NPTX2", "NPTXR", "NPTX1", "VGF", "SCG2", "NRN1"}

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # weight of the module in the rank blend


def _genes(col):
    return set(g.strip().upper() for g in str(col).split("|")[0].split(";") if g.strip())


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _base_scores(F_train, y_train, F_test):
    """Return (LR decision values, module score or None) for F_test."""
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill)
    Xte = F_test[keep].fillna(fill)
    sc = StandardScaler().fit(Xtr.values)
    Ztr = sc.transform(Xtr.values)
    Zte = sc.transform(Xte.values)

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    s_lr = clf.decision_function(Zte)

    # Prior-knowledge module: signs fixed a priori, no fitting.
    up_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_UP_PANEL]
    down_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_DOWN_PANEL]
    if len(up_idx) < 3:
        return s_lr, None
    module = Zte[:, up_idx].mean(axis=1)
    if len(down_idx) >= 2:
        module = module - Zte[:, down_idx].mean(axis=1)
    return s_lr, module


def _blend(s_lr, module, w):
    n = len(s_lr)
    return (1.0 - w) * rankdata(s_lr) / n + w * rankdata(module) / n


def _within_site_auc(y, s, sites):
    tot, acc = 0, 0.0
    for st in np.unique(sites):
        m = sites == st
        if len(np.unique(y[m])) < 2:
            continue
        acc += m.sum() * roc_auc_score(y[m], s[m])
        tot += m.sum()
    return acc / tot if tot else np.nan


def _sites(ctx, n):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if isinstance(meta, pd.DataFrame) and "site" in meta.columns and len(meta) == n:
        return meta["site"].astype(str).values
    return np.zeros(n, dtype=object).astype(str)


def _choose_weight(F_train, y, ctx):
    n = len(y)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == n else np.arange(n)
    sites = _sites(ctx, n)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 3:
        return 0.5
    oof_lr = np.full(n, np.nan)
    oof_mod = np.full(n, np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        s_lr, mod = _base_scores(F_train.iloc[tr], y[tr], F_train.iloc[te])
        if mod is None:
            return 0.5
        oof_lr[te] = s_lr
        oof_mod[te] = mod
    ok = ~np.isnan(oof_lr) & ~np.isnan(oof_mod)
    if ok.sum() < 10:
        return 0.5
    scores = []
    for w in WEIGHTS:
        s = _blend(oof_lr[ok], oof_mod[ok], w)
        scores.append(_within_site_auc(y[ok], s, sites[ok]))
    scores = np.nan_to_num(np.asarray(scores), nan=-1.0)
    best = scores.max()
    cands = [w for w, sc in zip(WEIGHTS, scores) if sc >= best - 1e-9]
    return min(cands, key=lambda w: abs(w - 0.5))           # tie-break toward equal blend


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    s_lr, module = _base_scores(F_train, y, F_test)
    if module is None:
        return s_lr
    w = _choose_weight(F_train, y, ctx)
    return _blend(s_lr, module, w)
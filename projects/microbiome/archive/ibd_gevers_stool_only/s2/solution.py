"""Genus-level relative abundance (prevalence-filtered) with location-wise normalization
(reference = training fold samples of the same sample_location) + location one-hot.
Both models use log relative abundance with a depth-aware pseudocount (half a read per sample,
i.e. zeros placed at each sample's own detection limit, so a zero in a deep sample ranks below
a zero in a shallow sample). RF (min_samples_leaf and max_features jointly tuned by grouped
inner CV) on location-wise percentiles of it; grouped-inner-CV-tuned L2 logistic regression on
the location-standardized version; probability average."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

LOCS = ["stool", "Terminal Ileum", "Rectum"]
DEPTH_COL = "__depth__"    # per-sample total read count, carried alongside the genus table
MIN_PREVALENCE = 0.10      # keep genera detected in >=10% of training samples
MIN_LOC_REF = 20           # min training samples of a location to use it as its own reference
LOG_PSEUDO = 1e-4          # fallback pseudocount if depth is unavailable
PSEUDO_READS = 0.5         # pseudocount in reads -> relative pseudocount 0.5 / depth per sample
LR_CS = np.logspace(-3, 1, 9)
W_RF = 0.5                 # weight of RF in the probability average
RF_LEAVES = [1, 3, 5, 10]  # candidate min_samples_leaf, chosen by grouped inner CV
RF_MAX_FEATURES = ["sqrt", 0.33]  # candidate max_features, chosen jointly with min_samples_leaf
RF_TUNE_TREES = 300
RF_FINAL_TREES = 1000


def featurize(items, ctx):
    depth = items.sum(axis=1)
    rel = items.div(depth.clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    G[DEPTH_COL] = depth.values.astype(float)
    return G


def _split_depth(F):
    if DEPTH_COL in F.columns:
        return F.drop(columns=DEPTH_COL), F[DEPTH_COL].values.astype(float)
    return F, None


def _locs(F, meta):
    """sample_location labels aligned to rows of F (by index if possible, else by position)."""
    if meta is None or "sample_location" not in getattr(meta, "columns", []):
        return np.array(["unknown"] * len(F))
    if meta.index.isin(F.index).all() and F.index.isin(meta.index).all() and len(meta) == len(F):
        loc = meta["sample_location"].reindex(F.index)
    else:
        loc = pd.Series(meta["sample_location"].values, index=F.index)
    return loc.astype(str).values


def _loc_onehot(loc):
    oh = np.zeros((len(loc), len(LOCS) + 1))
    for i, l in enumerate(loc):
        j = LOCS.index(l) if l in LOCS else len(LOCS)
        oh[i, j] = 1.0
    return oh


def _percentile(ref, X):
    """Mid-rank percentile of each value of X against the per-column reference matrix ref."""
    n = ref.shape[0]
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        s = np.sort(ref[:, j])
        lo = np.searchsorted(s, X[:, j], side="left")
        hi = np.searchsorted(s, X[:, j], side="right")
        out[:, j] = (lo + hi) / (2.0 * n)
    return out


def _zscore(ref, X):
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X - mu) / sd


def _loc_normalize(Xtr, loc_tr, X, loc, fn=_percentile):
    out = np.empty_like(X, dtype=float)
    for l in np.unique(loc):
        m = loc == l
        ref_mask = loc_tr == l
        ref = Xtr[ref_mask] if ref_mask.sum() >= MIN_LOC_REF else Xtr
        out[m] = fn(ref, X[m])
    return out


def _log_rel(X, depth):
    """log10 relative abundance with a per-sample pseudocount of PSEUDO_READS reads."""
    if depth is None:
        return np.log10(X + LOG_PSEUDO)
    pseudo = PSEUDO_READS / np.clip(depth, 1.0, None)
    return np.log10(X + pseudo[:, None])


def _inner_splits(y, groups):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= 5:
        return list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def _tune_rf(P, y, splits):
    """Pick (min_samples_leaf, max_features) maximizing pooled out-of-fold AUROC over grouped
    inner folds."""
    best, best_auc = (RF_LEAVES[0], RF_MAX_FEATURES[0]), -np.inf
    for mf in RF_MAX_FEATURES:
        for leaf in RF_LEAVES:
            oof = np.full(len(y), np.nan)
            for tr, va in splits:
                if len(np.unique(y[tr])) < 2:
                    continue
                m = RandomForestClassifier(n_estimators=RF_TUNE_TREES, min_samples_leaf=leaf,
                                           max_features=mf, random_state=0,
                                           n_jobs=-1).fit(P[tr], y[tr])
                oof[va] = m.predict_proba(P[va])[:, 1]
            ok = ~np.isnan(oof)
            if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
                continue
            auc = roc_auc_score(y[ok], oof[ok])
            if auc > best_auc:
                best, best_auc = (leaf, mf), auc
    return best


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    G_train, depth_tr = _split_depth(F_train)
    G_test, depth_te = _split_depth(F_test)
    if depth_tr is None or depth_te is None:
        depth_tr = depth_te = None
    prev = (G_train > 0).mean(axis=0)                       # label-free prevalence on training fold
    cols = G_train.columns[prev.values >= MIN_PREVALENCE]
    if len(cols) == 0:
        cols = G_train.columns
    Xtr, Xte = G_train[cols].values, G_test.reindex(columns=cols, fill_value=0).values
    loc_tr = _locs(F_train, ctx.get("meta_train"))
    loc_te = _locs(F_test, ctx.get("meta_test"))
    oh_tr, oh_te = _loc_onehot(loc_tr), _loc_onehot(loc_te)

    Ltr_raw, Lte_raw = _log_rel(Xtr, depth_tr), _log_rel(Xte, depth_te)
    Ptr = np.hstack([_loc_normalize(Ltr_raw, loc_tr, Ltr_raw, loc_tr), oh_tr])
    Pte = np.hstack([_loc_normalize(Ltr_raw, loc_tr, Lte_raw, loc_te), oh_te])

    Ltr = np.hstack([_loc_normalize(Ltr_raw, loc_tr, Ltr_raw, loc_tr, fn=_zscore), oh_tr])
    Lte = np.hstack([_loc_normalize(Ltr_raw, loc_tr, Lte_raw, loc_te, fn=_zscore), oh_te])

    groups = ctx.get("groups_train")
    groups = None if groups is None else np.asarray(groups)
    splits = _inner_splits(y, groups)

    leaf, mf = _tune_rf(Ptr, y, splits)
    rf = RandomForestClassifier(n_estimators=RF_FINAL_TREES, min_samples_leaf=leaf,
                                max_features=mf, random_state=0, n_jobs=-1)
    rf.fit(Ptr, y)
    p_rf = rf.predict_proba(Pte)[:, 1]

    sc = StandardScaler().fit(Ltr)
    lr = LogisticRegressionCV(Cs=LR_CS, cv=splits, penalty="l2",
                              scoring="roc_auc", solver="lbfgs", max_iter=5000, n_jobs=-1)
    lr.fit(sc.transform(Ltr), y)
    p_lr = lr.predict_proba(sc.transform(Lte))[:, 1]

    return W_RF * p_rf + (1.0 - W_RF) * p_lr
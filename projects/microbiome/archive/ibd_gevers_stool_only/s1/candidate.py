"""Genus-level relative abundance (unclassified genera resolved to deepest named rank),
per-location percentile normalization (reference = training samples of the same location)
+ sample-location covariate + healthy-reference Bray-Curtis dysbiosis scores
(out-of-fold for training samples) + extremely randomized trees."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

LOCS = ["stool", "terminal ileum", "rectum"]
MIN_REF = 20                                                # min training samples for a location reference
N_INNER = 5                                                 # inner grouped folds for out-of-fold dysbiosis scores


def _genus_label(col):
    parts = [p.strip() for p in col.split(";")]
    ranks = parts[:6]                                       # k, p, c, o, f, g
    if len(ranks) > 5 and len(ranks[5]) > 3:                # named genus, e.g. g__Blautia
        return ranks[5]
    for p in reversed(ranks[:5]):                           # deepest named higher rank
        if len(p) > 3:
            return p + "|unclassified"
    return "unclassified"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [_genus_label(c) for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _loc_labels(meta, n):
    if meta is None or "sample_location" not in getattr(meta, "columns", []):
        return np.array(["unknown"] * n)
    return meta["sample_location"].astype(str).str.strip().str.lower().values


def _loc_onehot(loc):
    return np.column_stack([(loc == l).astype(float) for l in LOCS])


def _midrank(ref, X):
    """Mid-rank percentile of each value of X relative to the reference rows, per column."""
    out = np.empty_like(X, dtype=float)
    n = ref.shape[0]
    for j in range(X.shape[1]):
        s = np.sort(ref[:, j])
        lo = np.searchsorted(s, X[:, j], side="left")
        hi = np.searchsorted(s, X[:, j], side="right")
        out[:, j] = (lo + hi) / (2.0 * n)
    return out


def _loc_percentile(Xtr, loc_tr, Xte, loc_te):
    Ptr, Pte = np.empty_like(Xtr, dtype=float), np.empty_like(Xte, dtype=float)
    for l in np.unique(np.concatenate([loc_tr, loc_te])):
        ref_mask = loc_tr == l
        ref = Xtr[ref_mask] if ref_mask.sum() >= MIN_REF else Xtr
        mtr, mte = loc_tr == l, loc_te == l
        if mtr.any():
            Ptr[mtr] = _midrank(ref, Xtr[mtr])
        if mte.any():
            Pte[mte] = _midrank(ref, Xte[mte])
    return Ptr, Pte


def _dysbiosis(Rref, yref, loc_ref, Rq, loc_q):
    """Median Bray-Curtis distance of query samples to reference controls / cases of the same
    location; returns [d_control, d_control - d_case]."""
    out = np.zeros((Rq.shape[0], 2))
    for l in np.unique(loc_q):
        mq = loc_q == l
        same = loc_ref == l
        ref_mask = same if same.sum() >= MIN_REF else np.ones_like(same)
        ctrl = ref_mask & (yref == 0)
        case = ref_mask & (yref == 1)
        if ctrl.sum() == 0:
            ctrl = yref == 0
        if case.sum() == 0:
            case = yref == 1
        d_ctrl = np.median(cdist(Rq[mq], Rref[ctrl], metric="braycurtis"), axis=1)
        d_case = np.median(cdist(Rq[mq], Rref[case], metric="braycurtis"), axis=1)
        out[mq, 0] = d_ctrl
        out[mq, 1] = d_ctrl - d_case
    return out


def _dysbiosis_features(Rtr, y, loc_tr, groups, Rte, loc_te):
    Dtr = np.zeros((Rtr.shape[0], 2))
    n_splits = min(N_INNER, len(np.unique(groups)))
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(Rtr, y, groups):
        Dtr[va_idx] = _dysbiosis(Rtr[tr_idx], y[tr_idx], loc_tr[tr_idx],
                                 Rtr[va_idx], loc_tr[va_idx])
    Dte = _dysbiosis(Rtr, y, loc_tr, Rte, loc_te)
    return Dtr, Dte


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    loc_tr = _loc_labels(ctx.get("meta_train"), Xtr.shape[0])
    loc_te = _loc_labels(ctx.get("meta_test"), Xte.shape[0])
    groups = ctx.get("groups_train")
    groups = np.arange(Xtr.shape[0]) if groups is None else np.asarray(groups)

    Rtr = Xtr / np.clip(Xtr.sum(axis=1, keepdims=True), 1e-12, None)
    Rte = Xte / np.clip(Xte.sum(axis=1, keepdims=True), 1e-12, None)
    Dtr, Dte = _dysbiosis_features(Rtr, y, loc_tr, groups, Rte, loc_te)

    Ptr, Pte = _loc_percentile(Xtr, loc_tr, Xte, loc_te)
    Xtr = np.hstack([Ptr, Dtr, _loc_onehot(loc_tr)])
    Xte = np.hstack([Pte, Dte, _loc_onehot(loc_te)])
    et = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt",
                              random_state=0, n_jobs=-1)
    et.fit(Xtr, y)
    return et.predict_proba(Xte)[:, 1]
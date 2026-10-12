"""Genus-level relative abundance (unclassified genera resolved to deepest named rank),
per-location percentile normalization (reference = training samples of the same location)
+ sample-location covariate + extremely randomized trees."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

LOCS = ["stool", "terminal ileum", "rectum"]
MIN_REF = 20                                                # min training samples for a location reference


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


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    loc_tr = _loc_labels(ctx.get("meta_train"), Xtr.shape[0])
    loc_te = _loc_labels(ctx.get("meta_test"), Xte.shape[0])
    Ptr, Pte = _loc_percentile(Xtr, loc_tr, Xte, loc_te)
    Xtr = np.hstack([Ptr, _loc_onehot(loc_tr)])
    Xte = np.hstack([Pte, _loc_onehot(loc_te)])
    et = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt",
                              random_state=0, n_jobs=-1)
    et.fit(Xtr, y_train)
    return et.predict_proba(Xte)[:, 1]
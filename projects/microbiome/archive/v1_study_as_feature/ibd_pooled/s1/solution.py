"""Genus-level log relative abundance censored at a common (cross-study) detection limit + alpha diversity
computed on floor-censored (pseudo-rarefied) profiles, z-scored within study (batch correction) + study one-hot;
ExtraTrees + CatBoost probability average."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None or "study" not in meta:
        return np.array(["all"] * len(items))
    s = meta["study"]
    try:
        if set(items.index).issubset(set(s.index)):
            return s.reindex(items.index).astype(str).values
    except Exception:
        pass
    return s.astype(str).values[: len(items)]


def _zscore_within(X, study):
    Z = np.array(X, dtype=float, copy=True)                 # writable copy
    for st in np.unique(study):
        m = study == st
        block = X[m]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z[m] = (block - mu) / sd
    return Z


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P), 0.0)
    return -t.sum(axis=1)


def _censor_renorm(P, floor):
    # drop abundances the shallowest-common depth could not resolve, then re-close the composition
    Q = np.where(P >= floor, P, 0.0)
    s = Q.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return Q / s


def featurize(items, ctx):
    depth = items.sum(axis=1).clip(lower=1)
    rel = items.div(depth, axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    # common detection limit: abundances below what shallow samples can resolve are censored,
    # so "absent" / "near-floor" means the same thing in every study (label-free, depth-only)
    floor = float(np.clip(np.quantile(1.0 / depth.values.astype(float), 0.9), 1e-5, 1e-3))
    Gv = G.values.astype(float)
    L = np.log10(np.maximum(Gv, floor))

    study = _study_labels(items, ctx)
    Z = _zscore_within(L, study)

    # alpha diversity on floor-censored profiles (pseudo-rarefaction to a common resolution)
    P_otu = _censor_renorm(rel.values.astype(float), floor)
    P_gen = _censor_renorm(Gv, floor)
    div = np.column_stack([
        _shannon(P_otu),
        _shannon(P_gen),
        np.log(1.0 / np.clip((P_gen ** 2).sum(axis=1), 1e-12, None)),   # log inverse Simpson
        np.log1p((P_otu > 0).sum(axis=1)),                               # OTU richness above common floor
        np.log1p((P_gen > 0).sum(axis=1)),                               # genus richness above common floor
    ])
    Dz = _zscore_within(div, study)

    F = pd.DataFrame(Z, index=items.index, columns=[f"z_{g}" for g in G.columns])
    for j, name in enumerate(["shannon_otu", "shannon_gen", "invsimpson_gen", "rich_otu", "rich_gen"]):
        F[f"div_{name}"] = Dz[:, j]
    for st in np.unique(study):
        F[f"study_{st}"] = (study == st).astype(float)
    return F


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test columns to training columns
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train).astype(int)

    # randomized thresholds avoid locking onto study-specific zero-floor boundaries
    et = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=2, max_features="sqrt",
                              random_state=0, n_jobs=-1)
    et.fit(Xtr, y)
    p_et = et.predict_proba(Xte)[:, 1]

    # shallow, strongly regularized boosting captures taxon x study / diversity interactions
    cb = CatBoostClassifier(iterations=800, learning_rate=0.03, depth=4, l2_leaf_reg=5.0,
                            rsm=0.3, subsample=0.8, bootstrap_type="Bernoulli",
                            random_seed=0, verbose=False, thread_count=-1,
                            allow_writing_files=False)
    cb.fit(Xtr, y)
    p_cb = cb.predict_proba(Xte)[:, 1]

    return 0.5 * p_et + 0.5 * p_cb
"""Genus-level log relative abundance censored at a common (cross-study) detection limit + alpha diversity
computed on truly rarefied OTU counts (common depth, averaged over several draws), z-scored within study
(batch correction) + study one-hot; ExtraTrees + CatBoost probability average."""
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


def _shannon_vec(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _div_metrics(otu_counts, gen_counts):
    n = otu_counts.sum()
    if n <= 0:
        return np.zeros(5)
    po = otu_counts / n
    pg = gen_counts / n
    return np.array([
        _shannon_vec(po),
        _shannon_vec(pg),
        np.log(1.0 / max(float((pg ** 2).sum()), 1e-12)),   # log inverse Simpson (genus)
        np.log1p(float((otu_counts > 0).sum())),            # OTU richness at common depth
        np.log1p(float((gen_counts > 0).sum())),            # genus richness at common depth
    ])


def _rarefied_diversity(C, gcodes, n_gen, D, n_draws=5, seed=0):
    # subsample each sample's reads without replacement to a common depth D (classic rarefaction),
    # average the diversity metrics over draws; samples shallower than D are used as-is
    rng = np.random.default_rng(seed)
    out = np.zeros((C.shape[0], 5))
    for i in range(C.shape[0]):
        row = C[i]
        nz = np.flatnonzero(row > 0)
        cnt = row[nz]
        gc = gcodes[nz]
        tot = int(cnt.sum())
        if tot <= D or nz.size == 0:
            out[i] = _div_metrics(cnt.astype(float),
                                  np.bincount(gc, weights=cnt, minlength=n_gen))
            continue
        acc = np.zeros(5)
        for _ in range(n_draws):
            sub = rng.multivariate_hypergeometric(cnt, D, method="marginals").astype(float)
            acc += _div_metrics(sub, np.bincount(gc, weights=sub, minlength=n_gen))
        out[i] = acc / n_draws
    return out


def featurize(items, ctx):
    depth = items.sum(axis=1).clip(lower=1)
    rel = items.div(depth, axis=0)
    genus = np.array([c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # common detection limit: abundances below what shallow samples can resolve are censored,
    # so "absent" / "near-floor" means the same thing in every study (label-free, depth-only)
    floor = float(np.clip(np.quantile(1.0 / depth.values.astype(float), 0.9), 1e-5, 1e-3))
    Gv = G.values.astype(float)
    L = np.log10(np.maximum(Gv, floor))

    study = _study_labels(items, ctx)
    Z = _zscore_within(L, study)

    # alpha diversity on rarefied counts at the common depth D = 1/floor
    C = np.rint(np.clip(items.values.astype(float), 0, None)).astype(np.int64)
    uniq, gcodes = np.unique(genus, return_inverse=True)
    D = int(round(1.0 / floor))
    div = _rarefied_diversity(C, gcodes, len(uniq), D)
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
"""Genus-level relative abundance + alpha-diversity (Shannon, inv-Simpson, rarefied richness at OTU/genus/family),
all percentile-normalized within study (label-free batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta["study"].reindex(items.index).astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return None


def _level(cols, k, empty):
    out = []
    for c in cols:
        p = c.split(";")
        out.append(p[k] if len(p) > k else empty)
    return np.array(out)


def _alpha(counts, genus_idx, family_idx, studies, n_gen, n_fam):
    """Shannon / inverse Simpson on full profile; richness after per-study rarefaction."""
    n = counts.shape[0]
    depth = counts.sum(axis=1)
    rel = counts / np.clip(depth, 1, None)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -np.nansum(np.where(rel > 0, rel * np.log(rel), 0.0), axis=1)
    simpson = (rel ** 2).sum(axis=1)
    inv_simpson = 1.0 / np.clip(simpson, 1e-12, None)

    rng = np.random.default_rng(0)
    rich_otu = np.zeros(n)
    rich_gen = np.zeros(n)
    rich_fam = np.zeros(n)
    labs = studies if studies is not None else np.array(["all"] * n)
    for s in pd.unique(labs):
        m = np.where(labs == s)[0]
        d = int(max(np.percentile(depth[m], 5), 1))
        for i in m:
            c = counts[i]
            if depth[i] > d:
                c = rng.multivariate_hypergeometric(c, d)
            present = c > 0
            rich_otu[i] = present.sum()
            rich_gen[i] = np.unique(genus_idx[present]).size
            rich_fam[i] = np.unique(family_idx[present]).size
    return np.column_stack([shannon, inv_simpson, rich_otu, rich_gen, rich_fam])


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = _level(rel.columns, 5, "g__")
    family = _level(rel.columns, 4, "f__")
    G = rel.T.groupby(genus).sum().T                        # samples x genera
    studies = _study_labels(items, ctx)

    counts = np.rint(items.fillna(0).values).clip(min=0).astype(np.int64)
    _, g_idx = np.unique(genus, return_inverse=True)
    _, f_idx = np.unique(family, return_inverse=True)
    A = _alpha(counts, g_idx, f_idx, studies, g_idx.max() + 1, f_idx.max() + 1)
    A = pd.DataFrame(A, index=items.index,
                     columns=["alpha_shannon", "alpha_invsimpson", "alpha_rich_otu",
                              "alpha_rich_genus", "alpha_rich_family"])
    F = pd.concat([G, A], axis=1)

    if studies is None:
        return F.rank(pct=True)
    out = F.copy()
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = F.loc[m].rank(axis=0, pct=True, method="average").values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
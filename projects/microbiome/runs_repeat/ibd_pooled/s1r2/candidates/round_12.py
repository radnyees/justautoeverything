"""Multi-level lineage-prefix relative abundance (cross-study prevalence-filtered) + alpha diversity,
per-study percentile-rank harmonization + rank-averaged ensemble of extremely randomized trees and
an L2 logistic regression on inverse-normal rank scores, both with study x class balanced weights."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata, norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=items.index)
    try:
        s = meta.loc[items.index, "study"]
        return pd.Series(s.values, index=items.index).astype(str)
    except Exception:
        if len(meta) == len(items):
            return pd.Series(meta["study"].values, index=items.index).astype(str)
        return pd.Series("all", index=items.index)


def _train_studies(F_train, ctx):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.array(["all"] * len(F_train))
    try:
        return meta.loc[F_train.index, "study"].astype(str).values
    except Exception:
        if len(meta) == len(F_train):
            return meta["study"].astype(str).values
        return np.array(["all"] * len(F_train))


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)


def _rarefied_richness(C, depth=1000):
    """Expected number of observed OTUs when subsampling n reads without replacement."""
    C = np.floor(np.asarray(C, dtype=float))
    N = C.sum(axis=1)
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        Ni = C[i][C[i] > 0]
        tot = N[i]
        if tot <= 0:
            continue
        n = min(depth, tot)
        # P(OTU absent) = C(tot - Ni, n) / C(tot, n); zero when tot - Ni < n
        ok = (tot - Ni) >= n
        logp = np.full(Ni.shape, -np.inf)
        a = tot - Ni[ok]
        logp[ok] = (gammaln(a + 1) - gammaln(a - n + 1)
                    - gammaln(tot + 1) + gammaln(tot - n + 1))
        out[i] = np.sum(1.0 - np.exp(logp))
    return out


def _lineage_prefix(col, level):
    """Lineage truncated at `level` (1=phylum ... 5=genus), dropping trailing empty ranks so
    unclassified reads are attributed to their deepest named ancestor."""
    parts = col.split(";")[: level + 1]
    while len(parts) > 1 and parts[-1].strip().endswith("__"):
        parts = parts[:-1]
    return ";".join(parts)


def featurize(items, ctx):
    counts = items.values.astype(float)
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    study = _study_labels(items, ctx)

    # taxonomy-aware aggregation at phylum..genus; names keep the full lineage prefix
    blocks = []
    for level, tag in [(1, "p"), (2, "c"), (3, "o"), (4, "f"), (5, "g")]:
        keys = np.array([_lineage_prefix(c, level) for c in rel.columns])
        B = rel.T.groupby(keys).sum().T
        B.columns = [f"{tag}|{k}" for k in B.columns]
        blocks.append(B)
    T = pd.concat(blocks, axis=1)
    # drop clades identical to their parent (single-child chains) and all-zero columns
    T = T.loc[:, (T > 0).any(axis=0)]
    T = T.loc[:, ~T.T.duplicated(keep="last").values]

    # cross-study prevalence filter (label-free): keep clades detected in >=5% of samples of
    # EVERY study, so retained features are measurable under all protocols/regions/depths
    prev = (T > 0).groupby(study.values).mean()
    keep = (prev.min(axis=0) >= 0.05).values
    if keep.sum() >= 20:
        T = T.loc[:, keep]

    genus_keys = np.array([_lineage_prefix(c, 5) for c in rel.columns])
    Gv = rel.T.groupby(genus_keys).sum().T.values

    # alpha diversity (IBD: reduced diversity is a robust cross-cohort signal)
    P = rel.values
    A = pd.DataFrame({
        "alpha__shannon_otu": _shannon(P),
        "alpha__shannon_genus": _shannon(Gv),
        "alpha__invsimpson_otu": 1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None),
        "alpha__rich_rare1000": _rarefied_richness(counts, 1000),
    }, index=items.index)
    X = pd.concat([T, A], axis=1)

    # per-study percentile ranks: label-free harmonization of study-specific scale shifts
    R = X.groupby(study.values).rank(pct=True, method="average")
    return R.reindex(index=X.index, columns=X.columns).fillna(0.5)


def _fit_logreg(Xtr, y, w, Xte, groups):
    """L2 logistic regression on inverse-normal (probit) rank scores; C by inner grouped CV.
    Normal scores stretch the tails of each study's distribution, where dysbiosis lives
    (blooms of facultative anaerobes, near-total loss of butyrate producers)."""
    Ztr = norm.ppf(np.clip(Xtr, 0.01, 0.99))
    Zte = norm.ppf(np.clip(Xte, 0.01, 0.99))
    Cs = np.logspace(-4, 0, 9)
    n_groups = len(np.unique(groups)) if groups is not None else 0
    try:
        if n_groups < 3:
            raise ValueError("not enough groups")
        splits = list(GroupKFold(n_splits=min(5, n_groups)).split(Ztr, y, groups))
        lr = LogisticRegressionCV(Cs=Cs, cv=splits, penalty="l2", scoring="roc_auc",
                                  solver="lbfgs", max_iter=2000, n_jobs=-1)
        lr.fit(Ztr, y, sample_weight=w)
    except Exception:
        lr = LogisticRegression(C=0.003, penalty="l2", solver="lbfgs", max_iter=2000)
        lr.fit(Ztr, y, sample_weight=w)
    return lr.decision_function(Zte)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    y = np.asarray(y_train)

    # study x class balanced weights: each study contributes equally, and within each study
    # cases and controls contribute equally -> splits target within-study discrimination
    studies = _train_studies(F_train, ctx)
    strata = pd.Series([f"{s}|{c}" for s, c in zip(studies, y)])
    n_strata = strata.map(strata.value_counts()).values.astype(float)
    n_cls = pd.Series(y).groupby(studies).nunique().reindex(studies).values.astype(float)
    w = 1.0 / (n_strata * n_cls)
    w = w * len(w) / w.sum()

    # extremely randomized trees: random cut-points spread importance across correlated clades
    # and give smoother, less study-idiosyncratic scores than optimized RF splits
    et = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", random_state=0, n_jobs=-1)
    et.fit(Xtr, y, sample_weight=w)
    p_et = et.predict_proba(Xte)[:, 1]

    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else None
    p_lr = _fit_logreg(Xtr, y, w, Xte, groups)

    # rank-average: the two models' scores live on different scales
    n = len(p_et)
    return 0.5 * (rankdata(p_et) / n) + 0.5 * (rankdata(p_lr) / n)
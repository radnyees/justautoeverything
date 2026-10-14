"""Multi-level taxonomy CLR (genus + family + order + class + phylum) + alpha diversity
(Shannon, Simpson, rarefied richness) + known IBD marker log-ratios (Gevers dysbiosis index,
butyrate producers vs Enterobacteriaceae, Proteobacteria vs Firmicutes), z-scored within each
study (batch harmonization), + rank-averaged ensemble of random forest and L2 logistic
regression (C tuned by grouped inner CV). Training samples are weighted so that every
(study, class) cell carries equal total weight (within-study balanced learning)."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold


def _study_labels(items, ctx, key="meta"):
    meta = ctx.get(key, None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].astype(str).values
        return np.zeros(len(items), dtype=object)


def _shannon(mat):
    tot = mat.sum(axis=1, keepdims=True)
    p = mat / np.clip(tot, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return h, p


def _expected_richness(counts_row, D):
    """Expected number of OTUs observed when rarefying a sample to depth D (analytical)."""
    n = counts_row[counts_row > 0]
    N = n.sum()
    if N <= D or D <= 0:
        return float(len(n))
    logC_N = gammaln(N + 1) - gammaln(D + 1) - gammaln(N - D + 1)
    rem = N - n
    absent = np.zeros_like(n, dtype=float)
    ok = rem >= D
    r = rem[ok]
    absent[ok] = np.exp(gammaln(r + 1) - gammaln(D + 1) - gammaln(r - D + 1) - logC_N)
    return float((1.0 - absent).sum())


def _zscore_by_study(X, studies):
    Z = np.zeros_like(X, dtype=float)
    for s in pd.unique(studies):
        m = studies == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z[m] = (X[m] - mu) / sd
    return Z


def _level_clr(counts, level, min_prev=0.02):
    """Aggregate counts to a taxonomy level (lineage prefix up to `level`), filter by
    prevalence, CLR with a depth-relative pseudocount."""
    lab = []
    for c in counts.columns:
        parts = c.split(";")
        lab.append(";".join(parts[: level + 1]) if len(parts) > level else c)
    g = counts.T.groupby(np.array(lab)).sum().T
    keep = (g > 0).mean(axis=0) >= min_prev
    g = g.loc[:, keep]
    rv = g.div(g.sum(axis=1).clip(lower=1), axis=0).values
    nz = np.where(rv > 0, rv, np.inf).min(axis=1, keepdims=True)
    nz[~np.isfinite(nz)] = 1e-6
    logx = np.log(rv + 0.5 * nz)
    clr = logx - logx.mean(axis=1, keepdims=True)
    names = [f"L{level}|{n}" for n in g.columns]
    return clr, names


def _marker_sum(counts_vals, cols_lower, tokens):
    """Total reads of OTUs whose lineage contains any of the given taxon tokens."""
    mask = np.array([any(t in c for t in tokens) for c in cols_lower])
    if not mask.any():
        return np.zeros(counts_vals.shape[0])
    return counts_vals[:, mask].sum(axis=1)


def _marker_features(counts):
    """Known IBD dysbiosis log-ratios (label-free, literature-defined)."""
    C = counts.values
    cols_lower = [c.lower() for c in counts.columns]
    depth = np.clip(C.sum(axis=1), 1, None)
    pc = 0.5  # half a read pseudocount

    # Gevers et al. 2014 microbial dysbiosis index
    up = _marker_sum(C, cols_lower, ["f__enterobacteriaceae", "f__pasteurellaceae",
                                     "f__veillonellaceae", "f__fusobacteriaceae"])
    down = _marker_sum(C, cols_lower, ["o__erysipelotrichales", "o__bacteroidales",
                                       "o__clostridiales"])
    md_index = np.log(up + pc) - np.log(down + pc)

    # butyrate producers vs Enterobacteriaceae
    buty = _marker_sum(C, cols_lower, ["g__faecalibacterium", "g__roseburia",
                                       "g__coprococcus", "g__eubacterium",
                                       "g__subdoligranulum", "g__anaerostipes"])
    entero = _marker_sum(C, cols_lower, ["f__enterobacteriaceae"])
    buty_ratio = np.log(buty + pc) - np.log(entero + pc)

    # Proteobacteria vs Firmicutes
    prot = _marker_sum(C, cols_lower, ["p__proteobacteria"])
    firm = _marker_sum(C, cols_lower, ["p__firmicutes"])
    pf_ratio = np.log(prot + pc) - np.log(firm + pc)

    # relative abundance of butyrate producers (log)
    buty_rel = np.log((buty + pc) / depth)

    F = np.column_stack([md_index, buty_ratio, pf_ratio, buty_rel])
    names = ["mk_dysbiosis_index", "mk_butyrate_vs_entero", "mk_proteo_vs_firmicutes",
             "mk_log_butyrate_rel"]
    return F, names


def featurize(items, ctx):
    counts = items.astype(float)
    studies = _study_labels(items, ctx)

    # genus-level (kept identical to before: genus name only)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in counts.columns]
    g = counts.T.groupby(np.array(genus)).sum().T              # samples x genera (counts)
    rel = g.div(g.sum(axis=1).clip(lower=1), axis=0)
    keep = (g > 0).mean(axis=0) >= 0.02
    g_f, rel_f = g.loc[:, keep], rel.loc[:, keep]
    rv = rel_f.values
    nz = np.where(rv > 0, rv, np.inf).min(axis=1, keepdims=True)
    nz[~np.isfinite(nz)] = 1e-6
    logx = np.log(rv + 0.5 * nz)
    clr = logx - logx.mean(axis=1, keepdims=True)

    # coarser taxonomy levels: family (4), order (3), class (2), phylum (1)
    clr_fam, fam_names = _level_clr(counts, 4)
    clr_ord, ord_names = _level_clr(counts, 3)
    clr_cls, cls_names = _level_clr(counts, 2)
    clr_phy, phy_names = _level_clr(counts, 1)

    # alpha diversity (OTU-level and genus-level)
    C = np.round(counts.values)
    h_otu, p_otu = _shannon(C)
    inv_simpson = 1.0 / np.clip((p_otu ** 2).sum(axis=1), 1e-12, None)
    h_gen, _ = _shannon(g.values)
    depth = C.sum(axis=1)
    rich = np.zeros(len(C))
    for s in pd.unique(studies):
        m = np.where(studies == s)[0]
        D = float(max(np.floor(np.quantile(depth[m], 0.1)), 100.0))
        for i in m:
            rich[i] = _expected_richness(C[i], D)
    div = np.column_stack([h_otu, np.log(inv_simpson), h_gen, np.log1p(rich)])
    div_names = ["alpha_shannon_otu", "alpha_log_invsimpson", "alpha_shannon_genus",
                 "alpha_log_rarefied_richness"]

    # known IBD marker log-ratios
    mk, mk_names = _marker_features(counts)

    X = np.hstack([clr, clr_fam, clr_ord, clr_cls, clr_phy, div, mk])
    Z = _zscore_by_study(X, studies)
    cols = (list(g_f.columns) + fam_names + ord_names + cls_names + phy_names
            + div_names + mk_names)
    return pd.DataFrame(Z, index=items.index, columns=cols)


def _inner_cv(y, groups, n_splits=5):
    if groups is not None and len(pd.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                .split(np.zeros(len(y)), y))


def _norm_rank(v):
    return (rankdata(v) - 0.5) / len(v)


def _study_class_weights(studies, y):
    """Equal total weight per (study, class) cell; normalized to mean 1."""
    keys = pd.Series([f"{s}|{c}" for s, c in zip(studies, y)])
    counts = keys.map(keys.value_counts()).values.astype(float)
    w = 1.0 / counts
    return w * (len(w) / w.sum())


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train)

    studies_tr = _study_labels(F_train, ctx, key="meta_train")
    sw = _study_class_weights(studies_tr, y)

    rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=sw)
    p_rf = rf.predict_proba(Xte)[:, 1]

    groups = ctx.get("groups_train", None)
    groups = np.asarray(groups) if groups is not None else None
    lr = LogisticRegressionCV(Cs=np.logspace(-3.5, -0.5, 7), penalty="l2",
                              cv=_inner_cv(y, groups), scoring="roc_auc",
                              solver="lbfgs", max_iter=3000, n_jobs=-1)
    lr.fit(Xtr, y, sample_weight=sw)
    p_lr = lr.predict_proba(Xte)[:, 1]

    return 0.5 * _norm_rank(p_rf) + 0.5 * _norm_rank(p_lr)
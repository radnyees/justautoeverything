"""Genus-level CLR + alpha diversity (Shannon, Simpson, rarefied richness), z-scored within
each study (batch harmonization), + random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
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


def featurize(items, ctx):
    counts = items.astype(float)
    studies = _study_labels(items, ctx)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in counts.columns]
    g = counts.T.groupby(np.array(genus)).sum().T              # samples x genera (counts)
    rel = g.div(g.sum(axis=1).clip(lower=1), axis=0)
    # mild prevalence filter: keep genera seen (>0) in at least 2% of samples
    keep = (g > 0).mean(axis=0) >= 0.02
    g_f, rel_f = g.loc[:, keep], rel.loc[:, keep]
    # CLR with a depth-relative pseudocount (half the min nonzero rel. abundance per sample)
    rv = rel_f.values
    nz = np.where(rv > 0, rv, np.inf).min(axis=1, keepdims=True)
    nz[~np.isfinite(nz)] = 1e-6
    logx = np.log(rv + 0.5 * nz)
    clr = logx - logx.mean(axis=1, keepdims=True)

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

    X = np.hstack([clr, div])
    Z = _zscore_by_study(X, studies)
    return pd.DataFrame(Z, index=items.index, columns=list(g_f.columns) + div_names)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    return rf.predict_proba(Xte)[:, 1]
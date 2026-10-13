"""Multi-level (genus + family + phylum) log relative abundance + alpha-diversity (Shannon, inverse
Simpson, Hurlbert rarefied richness at OTU and genus level), all standardized within study (batch
harmonization) + random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier

RARE_DEPTH = 2000
EXTRA_LEVELS = (4, 1)        # family, phylum (lineage indices: k=0 p=1 c=2 o=3 f=4 g=5)


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta.loc[items.index, "study"]
        return s.values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].values
        return np.zeros(len(items), dtype=object)


def _hurlbert(counts, depth):
    """Expected richness at min(depth, N) reads (exact rarefaction expectation, no randomness)."""
    out = np.zeros(counts.shape[0])
    for i in range(counts.shape[0]):
        x = counts[i]
        x = x[x > 0]
        N = x.sum()
        if N < 1:
            continue
        n = min(depth, N)
        rest = N - x
        ok = rest >= n
        logp = np.full(x.shape, -np.inf)
        r = rest[ok]
        logp[ok] = gammaln(r + 1) - gammaln(r - n + 1) - gammaln(N + 1) + gammaln(N - n + 1)
        out[i] = np.sum(1.0 - np.exp(logp))
    return out


def _alpha(counts, rel):
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(rel > 0, np.log(rel), 0.0)
    shannon = -(rel * lp).sum(axis=1)
    inv_simpson = 1.0 / np.clip((rel ** 2).sum(axis=1), 1e-12, None)
    rich = _hurlbert(counts, RARE_DEPTH)
    return shannon, np.log(inv_simpson), np.log1p(rich)


def _level_counts(counts_df, level, empty):
    names = []
    for c in counts_df.columns:
        parts = c.split(";")
        names.append(parts[level] if len(parts) > level else empty)
    return counts_df.T.groupby(np.array(names)).sum().T


def featurize(items, ctx):
    counts = np.round(np.clip(items.values.astype(float), 0, None))
    depth = counts.sum(axis=1)
    rel_otu = counts / np.clip(depth, 1, None)[:, None]
    cdf = pd.DataFrame(counts, index=items.index, columns=items.columns)

    Gc = _level_counts(cdf, 5, "g__")
    G = Gc.div(Gc.sum(axis=1).clip(lower=1), axis=0)          # samples x genera (relative)
    blocks = [np.log10(G.values + 1e-5)]
    names = list(G.columns)

    for lvl in EXTRA_LEVELS:
        prefix = "kpcofg"[lvl] + "__"
        Lc = _level_counts(cdf, lvl, prefix)
        R = Lc.div(Lc.sum(axis=1).clip(lower=1), axis=0)
        blocks.append(np.log10(R.values + 1e-5))
        names += ["lvl%d__%s" % (lvl, c) for c in R.columns]

    sh_o, si_o, ri_o = _alpha(counts, rel_otu)
    named = Gc.drop(columns=[c for c in Gc.columns if c == "g__"], errors="ignore")
    gcnt = named.values
    grel = gcnt / np.clip(gcnt.sum(axis=1), 1, None)[:, None]
    sh_g, si_g, ri_g = _alpha(gcnt, grel)
    D = np.column_stack([sh_o, si_o, ri_o, sh_g, si_g, ri_g])
    dnames = ["div__shannon_otu", "div__invsimpson_otu", "div__rich_otu",
              "div__shannon_genus", "div__invsimpson_genus", "div__rich_genus"]

    X = np.hstack(blocks + [D])
    studies = _study_labels(items, ctx)
    Z = np.zeros_like(X)
    for s in pd.unique(studies):
        m = studies == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (X[m] - mu) / sd
    return pd.DataFrame(Z, index=items.index, columns=names + dnames)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]
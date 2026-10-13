"""Multi-level (genus + family + phylum) log relative abundance + alpha-diversity (Shannon, inverse
Simpson, Hurlbert rarefied richness at OTU and genus level), all standardized within study (batch
harmonization); plus supervised beta-diversity features (distance of each sample to its own study's
training control / case centroids, leave-one-out for training items); rank-average of a random
forest and an inner-grouped-CV-tuned L2 logistic regression."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold

RARE_DEPTH = 2000
EXTRA_LEVELS = (4, 1)        # family, phylum (lineage indices: k=0 p=1 c=2 o=3 f=4 g=5)
LR_CS = np.logspace(-4, 0, 9)


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


def _meta_study(F, meta):
    """Study label per row of F from ctx meta_train / meta_test (index-aligned, else positional)."""
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(F), dtype=object)
    try:
        return meta.loc[F.index, "study"].astype(str).values
    except Exception:
        if len(meta) == len(F):
            return meta["study"].astype(str).values
        return np.zeros(len(F), dtype=object)


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


def _centroid_features(Ptr, y, s_tr, Pte, s_te):
    """Log Euclidean distance to own-study control and case centroids (LOO for training rows)."""
    p = Ptr.shape[1]
    glob = {c: (Ptr[y == c].sum(axis=0), int((y == c).sum())) for c in (0, 1)}
    stats = {}
    for s in pd.unique(s_tr):
        m = s_tr == s
        stats[s] = {c: (Ptr[m & (y == c)].sum(axis=0), int((m & (y == c)).sum())) for c in (0, 1)}

    def dist(x, s, c, own):
        S, n = stats.get(s, glob)[c]
        if own:
            S, n = S - x, n - 1
        if n < 2:
            S, n = glob[c]
            if own:
                S, n = S - x, n - 1
        cen = S / max(n, 1)
        return np.log(np.sqrt(np.sum((x - cen) ** 2) / p) + 1e-6)

    def build(P, st, yy):
        out = np.zeros((P.shape[0], 3))
        for i in range(P.shape[0]):
            d0 = dist(P[i], st[i], 0, yy is not None and yy[i] == 0)
            d1 = dist(P[i], st[i], 1, yy is not None and yy[i] == 1)
            out[i] = (d0, d1, d0 - d1)
        return out

    Btr = build(Ptr, s_tr, y)
    Bte = build(Pte, s_te, None)
    mu, sd = Btr.mean(axis=0), Btr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (Btr - mu) / sd, (Bte - mu) / sd


def _inner_splits(y, groups):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= 5:
        return list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train).astype(int)

    # supervised beta-diversity: distance to study-specific healthy / IBD centroids (taxon profile only)
    prof = np.array([not str(c).startswith("div__") for c in cols])
    mtr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    mte = ctx.get("meta_test") if isinstance(ctx, dict) else None
    s_tr, s_te = _meta_study(F_train, mtr), _meta_study(F_test, mte)
    Btr, Bte = _centroid_features(np.clip(Xtr[:, prof], -5, 5), y, s_tr,
                                  np.clip(Xte[:, prof], -5, 5), s_te)
    Xtr = np.hstack([Xtr, Btr])
    Xte = np.hstack([Xte, Bte])

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = None if groups is None else np.asarray(groups)
    Xtr_c = np.clip(Xtr, -5, 5)
    Xte_c = np.clip(Xte, -5, 5)
    lr = LogisticRegressionCV(Cs=LR_CS, cv=_inner_splits(y, groups), penalty="l2",
                              scoring="roc_auc", solver="lbfgs", max_iter=3000, n_jobs=-1)
    lr.fit(Xtr_c, y)
    p_lr = lr.predict_proba(Xte_c)[:, 1]

    n = len(p_rf)
    return 0.5 * (rankdata(p_rf) + rankdata(p_lr)) / n
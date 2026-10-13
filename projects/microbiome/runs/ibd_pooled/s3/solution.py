"""Lineage-aware multi-level (genus/family/order/class/phylum) relative abundance + alpha-diversity
+ Gevers microbial dysbiosis index (log-ratio); ensemble of a random forest and an L2
logistic regression on per-sample, per-level CLR (genus/family/order/class/phylum) abundances plus
rarefied genus presence/absence, with the linear design centred per study on that study's
training-fold controls. The two models' scores are rank-averaged WITHIN each test study
(matching the within-study metric)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RARE_DEPTH = 1000
CLR_PSEUDO = 1e-5          # relative-abundance pseudocount for the CLR transform
LR_MIN_PREV = 0.05         # taxon prevalence (training fold) required for the linear model
PA_MAX_PREV = 0.95         # presence features must vary in the training fold
LR_C = 0.05
LR_LEVELS = ("G|", "F|", "O|", "CL|", "PH|")   # each level is CLR-transformed as its own composition
MIN_REF = 5                # minimum reference samples for a per-study control centre

# Gevers et al. 2014 (Cell Host Microbe) microbial dysbiosis index taxa
MD_UP_FAMILIES = {"Enterobacteriaceae", "Pasteurellaceae", "Fusobacteriaceae",
                  "Neisseriaceae", "Veillonellaceae", "Gemellaceae"}
MD_DOWN_ORDERS = {"Bacteroidales", "Clostridiales"}
MD_DOWN_FAMILIES = {"Erysipelotrichaceae", "Bifidobacteriaceae"}


def _shannon(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _alpha(counts, genus_idx, n_genus, rng):
    """Alpha-diversity for one sample's raw OTU counts (int vector), plus the rarefied
    genus presence vector (depth-normalized detection)."""
    tot = counts.sum()
    out = {}
    if tot <= 0:
        return (dict(div_shannon_otu=0.0, div_simpson_otu=0.0, div_shannon_genus=0.0,
                     div_rich_otu=0.0, div_rich_genus=0.0, div_log_rich_obs=0.0),
                np.zeros(n_genus))
    p = counts / tot
    out["div_shannon_otu"] = _shannon(p)
    out["div_simpson_otu"] = float(1.0 - (p ** 2).sum())
    pg = np.bincount(genus_idx, weights=p, minlength=n_genus)
    out["div_shannon_genus"] = _shannon(pg)
    # rarefied richness (depth-normalized); samples below depth keep all reads
    if tot > RARE_DEPTH:
        sub = rng.multivariate_hypergeometric(counts.astype(np.int64), RARE_DEPTH)
    else:
        sub = counts
    present = sub > 0
    out["div_rich_otu"] = float(present.sum())
    out["div_rich_genus"] = float(np.unique(genus_idx[present]).size)
    out["div_log_rich_obs"] = float(np.log1p((counts > 0).sum()))
    pres_genus = (np.bincount(genus_idx, weights=sub.astype(float), minlength=n_genus) > 0)
    return out, pres_genus.astype(float)


def _level(col, prefix):
    for part in col.split(";"):
        part = part.strip()
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _lineage(col, depth):
    """Lineage key truncated to the first `depth` ranks (k,p,c,o,f,g -> depth 6)."""
    parts = [p.strip() for p in col.split(";")]
    parts = (parts + [""] * depth)[:depth]
    return ";".join(parts)


def _md_index(items):
    """Gevers dysbiosis index: log(sum up-taxa / sum down-taxa), pseudocount 0.5 read."""
    cols = list(items.columns)
    fam = np.array([_level(c, "f__") for c in cols])
    order = np.array([_level(c, "o__") for c in cols])
    up = np.isin(fam, list(MD_UP_FAMILIES))
    down = (np.isin(order, list(MD_DOWN_ORDERS)) | np.isin(fam, list(MD_DOWN_FAMILIES))) & ~up
    X = items.values.astype(float)
    depth = X.sum(axis=1).clip(min=1)
    eps = 0.5 / depth
    up_rel = X[:, up].sum(axis=1) / depth
    down_rel = X[:, down].sum(axis=1) / depth
    return pd.DataFrame({
        "md_index": np.log(up_rel + eps) - np.log(down_rel + eps),
        "md_log_up": np.log(up_rel + eps),
        "md_log_down": np.log(down_rel + eps),
    }, index=items.index)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    cols = list(rel.columns)
    # lineage-aware keys: an unclassified genus stays attached to its family/order
    genus = np.array(["G|" + _lineage(c, 6) for c in cols])
    family = np.array(["F|" + _lineage(c, 5) for c in cols])
    order = np.array(["O|" + _lineage(c, 4) for c in cols])
    klass = np.array(["CL|" + _lineage(c, 3) for c in cols])
    phylum = np.array(["PH|" + _lineage(c, 2) for c in cols])
    relT = rel.T
    G = relT.groupby(genus).sum().T                         # samples x genera
    Fm = relT.groupby(family).sum().T                       # samples x families
    O = relT.groupby(order).sum().T                         # samples x orders
    Cl = relT.groupby(klass).sum().T                        # samples x classes
    Ph = relT.groupby(phylum).sum().T                       # samples x phyla

    genus_names, genus_idx = np.unique(genus, return_inverse=True)
    n_genus = int(genus_idx.max()) + 1 if genus_idx.size else 1
    rng = np.random.default_rng(0)
    C = np.rint(items.values).astype(np.int64)
    C[C < 0] = 0
    rows, pres = [], []
    for i in range(C.shape[0]):
        d, pr = _alpha(C[i], genus_idx, n_genus, rng)
        rows.append(d)
        pres.append(pr)
    D = pd.DataFrame(rows, index=items.index)
    P = pd.DataFrame(np.vstack(pres) if pres else np.zeros((0, n_genus)), index=items.index,
                     columns=["P|" + str(g)[2:] for g in genus_names[:n_genus]])
    M = _md_index(items)
    return pd.concat([G, Fm, O, Cl, Ph, D, M, P], axis=1)


def _clr_block(F, tcols):
    B = np.log(F.reindex(columns=tcols, fill_value=0).values.astype(float) + CLR_PSEUDO)
    return B - B.mean(axis=1, keepdims=True)


def _linear_design(F, level_cols, ecols, pcols):
    """Per-sample CLR within each taxonomic level + engineered (diversity / MDI) features
    + rarefied genus presence/absence."""
    blocks = [_clr_block(F, tcols) for tcols in level_cols if len(tcols) > 1]
    blocks.append(F.reindex(columns=ecols, fill_value=0).values.astype(float))
    if pcols:
        blocks.append(F.reindex(columns=pcols, fill_value=0).values.astype(float))
    return np.hstack(blocks)


def _study(ctx, key, n):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []) or len(meta) != n:
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"]).astype(str)


def _control_centre(Ltr, Lte, y, s_tr, s_te):
    """Subtract, per study, the mean of that study's training-fold controls (harmonization:
    removes study offsets while preserving each study's case-vs-control shift)."""
    glob_ref = Ltr[y == 0] if (y == 0).sum() >= MIN_REF else Ltr
    glob_mu = glob_ref.mean(axis=0)
    Ltr, Lte = Ltr.copy(), Lte.copy()
    for s in np.unique(np.concatenate([s_tr, s_te])):
        in_tr = s_tr == s
        ref = Ltr[in_tr & (y == 0)]
        if ref.shape[0] < MIN_REF:
            ref = Ltr[in_tr]
        mu = ref.mean(axis=0) if ref.shape[0] >= MIN_REF else glob_mu
        Ltr[in_tr] -= mu
        Lte[s_te == s] -= mu
    return Ltr, Lte


def _within_group_rank(score, groups):
    """Rank scores within each group, scaled to (0, 1]."""
    out = np.empty(len(score), dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] = rankdata(score[m]) / m.sum()
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    cols = F_train.columns
    # random forest: abundances at all levels, diversity, MDI; no presence block
    rf_cols = [c for c in cols if not str(c).startswith("P|")]
    Xtr = F_train[rf_cols].values
    Xte = F_test.reindex(columns=rf_cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # linear model on compositional (CLR) profiles at each level; prevalence fitted
    # on the training fold
    level_cols = []
    for pref in LR_LEVELS:
        tcols = [c for c in cols if str(c).startswith(pref)]
        if not tcols:
            continue
        prev = (F_train[tcols].values > 0).mean(axis=0)
        level_cols.append([c for c, p in zip(tcols, prev) if p >= LR_MIN_PREV])
    ecols = [c for c in cols if str(c).startswith(("div_", "md_"))]
    pall = [c for c in cols if str(c).startswith("P|")]
    pprev = F_train[pall].values.mean(axis=0) if pall else np.array([])
    pcols = [c for c, p in zip(pall, pprev) if LR_MIN_PREV <= p <= PA_MAX_PREV]
    Ltr = _linear_design(F_train, level_cols, ecols, pcols)
    Lte = _linear_design(F_test, level_cols, ecols, pcols)
    s_tr = _study(ctx, "meta_train", Ltr.shape[0])
    s_te = _study(ctx, "meta_test", Lte.shape[0])
    Ltr, Lte = _control_centre(Ltr, Lte, y, s_tr, s_te)
    sc = StandardScaler().fit(Ltr)
    lr = LogisticRegression(C=LR_C, penalty="l2", max_iter=5000)
    lr.fit(sc.transform(Ltr), y)
    s_lr = lr.decision_function(sc.transform(Lte))

    # rank-average within each test study so both models carry equal weight in every study
    return 0.5 * (_within_group_rank(p_rf, s_te) + _within_group_rank(s_lr, s_te))
"""log2 + per-sample median centring on a FIXED reference set of proteins quantified in (nearly)
all samples (so the normalization factor is not biased by abundance-dependent missingness /
per-sample detection depth) + erythrocyte-contamination regression (label-free) +
half-minimum imputation; WITHIN-SITE weighted rank-average of (a) whole-proteome L2 logistic
regression (proteins quantified in >=70% of train), (b) SIGN-CONSTRAINED L2 logistic regression on
a prior-biology CSF AD protein panel (panel proteins kept down to 30% quantification; proteins with
a well-established AD direction are forced to carry a coefficient in that direction, the rest are
free), (c) L2 logistic regression on five prior-biology MODULE scores (PC1-sign-aligned members),
and (d) a fixed-weight YWHAG:NPTX2-style log-ratio score (up-in-AD glycolysis/14-3-3 minus
down-in-AD pentraxin/VGF synaptic proteins), whose only fitted quantity is its overall sign on the
training fold. The weight-free ratio gets double weight: it has no fitted coefficients, so its
ranking is the lowest-variance member at small n. Member scores are rank-normalized within each
test sample's site (label-free harmonization of score scales), matching the within-site metric."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# CSF AD proteins from the literature (Bader 2020; Higginbotham 2020; Johnson 2022; ...)
MODULES = {
    # glycolysis / 14-3-3 / astro-microglial metabolism (up in AD CSF)
    "glycolysis": {"SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "YWHAH", "PKM", "ALDOA",
                   "ALDOC", "MDH1", "GDA", "TMSB10", "GAPDH", "ENO1", "ENO2", "LDHA", "LDHB",
                   "PGK1", "TPI1", "PPIA", "PEBP1", "FABP3", "CALM1", "CALM2", "CALM3", "SOD1",
                   "PARK7"},
    # glial / inflammation
    "glial": {"CHI3L1", "CHI3L2", "CHIT1", "SPP1", "GFAP", "TREM2", "CD44", "VIM", "CD14"},
    # neuronal injury / tau-related
    "neuronal": {"NEFL", "NEFM", "NEFH", "MAPT", "NRGN", "UCHL1", "SNAP25", "SYN1", "STMN1",
                 "BASP1"},
    # synaptic / neurosecretory (down in AD CSF)
    "synaptic": {"NPTX2", "NPTX1", "NPTXR", "VGF", "SCG2", "SCG3", "SCG5", "CHGA", "CHGB", "PENK",
                 "PDYN", "CRH", "NRXN1", "NRXN2", "NRXN3", "CBLN4", "PCSK1", "SST", "NPY"},
    # amyloid / lipid / complement
    "complement": {"APP", "APOE", "CLU", "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH",
                   "SPARCL1"},
}
PANEL = set().union(*MODULES.values())

# literature direction in AD CSF used to constrain the panel model's coefficient signs;
# proteins not listed (complement/amyloid/lipid, SYN1, STMN1, BASP1) are left unconstrained
DIR_UP = (MODULES["glycolysis"] | MODULES["glial"]
          | {"NEFL", "NEFM", "NEFH", "MAPT", "NRGN", "UCHL1", "SNAP25"})
DIR_DOWN = MODULES["synaptic"]

# YWHAG:NPTX2-style ratio: numerator up in AD CSF, denominator down in AD CSF
RATIO_UP = {"YWHAG", "SMOC1", "PKM", "ALDOA"}
RATIO_DOWN = {"NPTX2", "NPTXR", "NPTX1", "VGF"}
RATIO_WEIGHT = 2.0        # weight-free member counts as two fitted members

# erythrocyte-specific proteins: blood-contamination quality markers (Bader 2020 style)
ERY = {"HBA1", "HBA2", "HBB", "HBD", "CA1", "PRDX2", "CAT", "BLVRB", "SLC4A1", "ALAS2", "SPTA1"}

MIN_PRESENT_ALL = 0.7     # whole-proteome model
MIN_PRESENT_PANEL = 0.3   # prior-biology panel: allow low-abundance markers
REF_PRESENT = (0.95, 0.9, 0.8)   # reference-set quantification thresholds (first with >=50)


def _genes(col):
    g = str(col).split("|")[0]
    return set(x.strip().upper() for x in g.split(";") if x.strip())


def _remove_contamination(L):
    """Regress a per-sample erythrocyte contamination score out of every other protein."""
    ery_cols = [c for c in L.columns if _genes(c) & ERY and L[c].notna().mean() >= 0.3]
    if len(ery_cols) < 2:
        return L
    E = L[ery_cols]
    E = E.fillna(E.min() - 1.0)
    Z = (E - E.mean()) / (E.std(ddof=0) + 1e-9)
    s = Z.mean(axis=1).values
    s = s - s.mean()
    out = L.copy()
    ery_set = set(ery_cols)
    for c in L.columns:
        if c in ery_set:
            continue
        x = L[c].values
        m = ~np.isnan(x)
        if m.sum() < 10:
            continue
        sm = s[m] - s[m].mean()
        den = np.dot(sm, sm)
        if den <= 1e-9:
            continue
        beta = np.dot(sm, x[m] - x[m].mean()) / den
        out[c] = x - beta * s                              # NaN stays NaN
    return out


def _reference_median(L):
    """Per-sample median over a fixed set of (nearly) always-quantified proteins, so every
    sample's normalization factor is computed from the same proteins (label-free)."""
    present = L.notna().mean()
    for thr in REF_PRESENT:
        ref = L.columns[present >= thr]
        if len(ref) >= 50:
            return L[ref].median(axis=1)
    return L.median(axis=1)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(_reference_median(L), axis=0)                # reference-set median centring
    return _remove_contamination(L)


def _lr(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def _direction(col):
    g = _genes(col)
    up, down = bool(g & DIR_UP), bool(g & DIR_DOWN)
    if up and not down:
        return 1.0
    if down and not up:
        return -1.0
    return 0.0


def _signed_lr(Xtr, ytr, Xte, signs, C):
    """L2 logistic regression (sklearn objective 0.5|w|^2 + C*sum logloss) with coefficients of
    direction-known features constrained to the literature sign; others unconstrained."""
    sc = StandardScaler().fit(Xtr)
    orient = np.where(signs != 0, signs, 1.0)
    Ztr = sc.transform(Xtr) * orient
    Zte = sc.transform(Xte) * orient
    yy = 2.0 * np.asarray(ytr, dtype=float) - 1.0
    p = Ztr.shape[1]

    def obj(theta):
        w, b = theta[:p], theta[p]
        m = yy * (Ztr @ w + b)
        loss = np.logaddexp(0.0, -m).sum()
        dz = -yy * expit(-m)
        g = np.empty_like(theta)
        g[:p] = w + C * (Ztr.T @ dz)
        g[p] = C * dz.sum()
        return 0.5 * np.dot(w, w) + C * loss, g

    bounds = [(0.0, None) if s != 0 else (None, None) for s in signs] + [(None, None)]
    res = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000})
    return Zte @ res.x[:p]


def _impute(F_train, F_test, cols):
    fill = F_train[cols].min() - 1.0                        # half-minimum on the log2 scale
    return F_train[cols].fillna(fill), F_test[cols].fillna(fill)


def _pc1_signs(Z):
    """Label-free sign of each member's loading on the module's first principal component."""
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C)
    w, V = np.linalg.eigh(C)
    v = V[:, -1]
    signs = np.where(v >= 0, 1.0, -1.0)
    if signs.sum() < 0:                                     # orient toward the majority
        signs = -signs
    return signs


def _module_scores(Ptr, Pte, panel_cols):
    """Mean of train-standardized, PC1-sign-aligned member proteins per prior-biology module."""
    mu = Ptr.mean()
    sd = Ptr.std(ddof=0) + 1e-9
    Ztr = (Ptr - mu) / sd
    Zte = (Pte - mu) / sd
    Mtr, Mte = [], []
    for genes in MODULES.values():
        cols = [c for c in panel_cols if _genes(c) & genes]
        if len(cols) >= 2:
            ztr = Ztr[cols].values
            zte = Zte[cols].values
            sg = _pc1_signs(ztr)
            Mtr.append((ztr * sg).mean(axis=1))
            Mte.append((zte * sg).mean(axis=1))
    if len(Mtr) < 2:
        return None, None
    return np.column_stack(Mtr), np.column_stack(Mte)


def _ratio_score(Ptr, Pte, panel_cols, y):
    """Fixed-weight log-ratio: mean z(up-in-AD) minus mean z(down-in-AD); sign checked on train."""
    up = [c for c in panel_cols if _genes(c) & RATIO_UP]
    down = [c for c in panel_cols if _genes(c) & RATIO_DOWN]
    if not up or not down:
        return None
    mu = Ptr.mean()
    sd = Ptr.std(ddof=0) + 1e-9
    Ztr = (Ptr - mu) / sd
    Zte = (Pte - mu) / sd
    rtr = Ztr[up].mean(axis=1).values - Ztr[down].mean(axis=1).values
    rte = Zte[up].mean(axis=1).values - Zte[down].mean(axis=1).values
    if len(np.unique(y)) == 2 and roc_auc_score(y, rtr) < 0.5:
        rte = -rte
    return rte


def _test_sites(ctx, n):
    """Site label per test sample (harmonization only); None if unavailable."""
    meta = ctx.get("meta_test") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "site" not in meta.columns:
                return None
            s = meta["site"].values
        else:
            s = np.asarray(pd.DataFrame(meta)["site"].values)
    except Exception:
        return None
    if len(s) != n:
        return None
    return np.asarray(s).astype(str)


def _site_rank(s, sites):
    """Percentile rank of scores within each site (pooled if sites unknown)."""
    s = np.asarray(s, dtype=float)
    if sites is None:
        return rankdata(s) / len(s)
    out = np.empty(len(s))
    for g in np.unique(sites):
        m = sites == g
        out[m] = rankdata(s[m]) / m.sum()
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    present = F_train.notna().mean()

    # (a) whole-proteome model
    keep = F_train.columns[present >= MIN_PRESENT_ALL]
    Xtr_df, Xte_df = _impute(F_train, F_test, keep)
    s_all = _lr(Xtr_df.values, y, Xte_df.values, C=0.05)
    sites = _test_sites(ctx, len(s_all))

    # (b) prior-biology panel model (relaxed presence filter; missingness ~ low abundance),
    #     coefficient signs constrained to the literature AD direction where it is established
    panel_cols = [c for c in F_train.columns[present >= MIN_PRESENT_PANEL] if _genes(c) & PANEL]
    if len(panel_cols) < 3:
        return s_all
    Ptr, Pte = _impute(F_train, F_test, panel_cols)
    signs = np.array([_direction(c) for c in panel_cols])
    s_pan = _signed_lr(Ptr.values, y, Pte.values, signs, C=0.1)
    ranks = [_site_rank(s_all, sites), _site_rank(s_pan, sites)]
    weights = [1.0, 1.0]

    # (c) module-score model: few aggregated, noise-averaged prior-biology features
    Mtr, Mte = _module_scores(Ptr, Pte, panel_cols)
    if Mtr is not None:
        s_mod = _lr(Mtr, y, Mte, C=0.5)
        ranks.append(_site_rank(s_mod, sites))
        weights.append(1.0)

    # (d) fixed-weight YWHAG:NPTX2-style ratio (no fitted weights) -> double weight
    s_rat = _ratio_score(Ptr, Pte, panel_cols, y)
    if s_rat is not None:
        ranks.append(_site_rank(s_rat, sites))
        weights.append(RATIO_WEIGHT)
    return np.average(np.vstack(ranks), axis=0, weights=np.asarray(weights))
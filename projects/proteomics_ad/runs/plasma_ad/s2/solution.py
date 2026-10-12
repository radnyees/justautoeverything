"""log2 + per-sample median centring + a-priori exclusion of immunoglobulin variable-region
groups (individual clonal repertoire, not disease biology) + batch-consistent detection filter +
per-batch protein median centring (train-fold stats) + half-minimum imputation + train-fold
rank-based inverse-normal transform (test mapped through the train ECDF; ties -> mid-rank, so
imputed values sit at the mean normal score of the bottom quantile rather than an arbitrary
extreme) + prior-weighted (known AD plasma biomarkers get a weaker penalty) L2 logistic
regression, probability-averaged with a compact L2 logistic regression restricted to the
a-priori AD biomarker panel (low-variance knowledge anchor).
The a-priori panel is extended with abundant plasma proteins that have replicated AD/amyloid
associations in the literature (fibrinogen, cystatin C, B2M, IGFBP2, gelsolin, vitronectin,
BCHE, SERPINA1, AGT, PON1, LRG1)."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CLIP = 3.0     # safety clip on the standardised normal scores (rarely active after INT)
MIN_DET = 0.7  # required detection rate within EVERY (sufficiently large) training batch
PRIOR_W = 2.0  # feature scale for prior-biology proteins (== 1/PRIOR_W^2 L2 penalty)
PANEL_C = 0.1  # L2 strength of the prior-panel-only model (fixed a priori)
PANEL_MIX = 0.5  # weight of the prior-panel model in the probability average
MIN_PANEL = 8  # need at least this many panel proteins to use the panel model

# immunoglobulin variable-region gene prefixes (person-specific antibody repertoire)
IG_VAR_PREFIXES = ("IGHV", "IGKV", "IGLV")

# fixed a-priori list of AD / neurodegeneration-associated proteins reported in plasma/CSF
PRIOR_GENES = {
    # apolipoproteins / lipid transport
    "APOE", "APOA1", "APOA2", "APOA4", "APOB", "APOC1", "APOC3", "APOD", "APOH", "APOM", "CLU",
    # complement cascade
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "C4BPA", "C5", "C9",
    "CFH", "CFB", "CFI", "CFD", "CR1",
    # acute phase / inflammation
    "SERPINA3", "A2M", "CP", "HP", "SAA1", "CRP", "ORM1", "VCAM1", "ICAM1",
    # transport / amyloid-tau related
    "TTR", "APP", "APLP2", "MAPT", "PPY", "SPARCL1",
    # neuro-glial / synaptic markers (detected at low level on deep plasma MS)
    "NEFL", "GFAP", "YWHAG", "SMOC1", "SPP1", "CHI3L1", "NPTX2", "VGF", "NRGN", "TREM2",
    # replicated abundant plasma AD / amyloid associations
    "FGA", "FGB", "FGG",            # fibrinogen: BBB leakage, Abeta-fibrin(ogen) binding
    "CST3",                         # cystatin C: co-deposits with Abeta, CSF/plasma marker
    "B2M",                          # beta-2-microglobulin: aging/cognitive decline factor
    "IGFBP2",                       # IGF binding protein 2: elevated with AD / atrophy
    "GSN",                          # gelsolin: Abeta-binding, reduced in AD plasma
    "VTN",                          # vitronectin: plaque-associated
    "BCHE",                         # butyrylcholinesterase: cholinergic, plaque-associated
    "SERPINA1",                     # alpha-1-antitrypsin: replicated plasma AD protein
    "AGT",                          # angiotensinogen: renin-angiotensin / vascular risk
    "PON1",                         # paraoxonase-1: HDL oxidative-stress, lower in AD
    "LRG1",                         # leucine-rich alpha-2-glycoprotein: neuroinflammation
}


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return pd.DataFrame(meta)["batch"].astype(str).values
    except Exception:
        return np.zeros(n, dtype=object)


def _genes(col):
    gene = str(col).split("|")[0]
    return [g.strip().upper() for g in gene.split(";") if g.strip()]


def _is_prior(col):
    return any(g in PRIOR_GENES for g in _genes(col))


def _is_ig_variable(col):
    gs = _genes(col)
    return len(gs) > 0 and all(g.startswith(IG_VAR_PREFIXES) for g in gs)


def _rank_int(Xtr, Xte):
    """Per-column rank-based inverse-normal transform fitted on train; test via train ECDF."""
    n = Xtr.shape[0]
    lo, hi = 0.5 / n, 1.0 - 0.5 / n
    Ztr = np.empty_like(Xtr, dtype=float)
    Zte = np.empty_like(Xte, dtype=float)
    for j in range(Xtr.shape[1]):
        x = Xtr[:, j]
        Ztr[:, j] = norm.ppf((rankdata(x, method="average") - 0.5) / n)
        s = np.sort(x)
        u = 0.5 * (np.searchsorted(s, Xte[:, j], side="left")
                   + np.searchsorted(s, Xte[:, j], side="right")) / n
        Zte[:, j] = norm.ppf(np.clip(u, lo, hi))
    return Ztr, Zte


def fit_predict(F_train, y_train, F_test, ctx):
    btr = _batches(ctx.get("meta_train"), len(F_train))
    bte = _batches(ctx.get("meta_test"), len(F_test))

    # a-priori exclusion of immunoglobulin variable-region groups (label-free)
    nonig = [c for c in F_train.columns if not _is_ig_variable(c)]
    if len(nonig) >= 50:
        F_train = F_train[nonig]
        F_test = F_test[nonig]

    # keep proteins quantified in >= MIN_DET of train samples overall AND within each batch,
    # so imputed blocks cannot encode acquisition batch
    det = F_train.notna()
    ok = det.mean() >= MIN_DET
    for b in np.unique(btr):
        mtr = btr == b
        if mtr.sum() < 5:
            continue
        ok &= det[mtr].mean() >= MIN_DET
    keep = F_train.columns[ok.values]
    if len(keep) < 50:                                      # safety fallback
        keep = F_train.columns[det.mean() >= MIN_DET]
    Tr = F_train[keep].copy()
    Te = F_test[keep].copy()

    # per-batch location correction: subtract each protein's within-batch median (train rows only)
    gmed = Tr.median()
    for b in np.unique(btr):
        mtr = btr == b
        if mtr.sum() < 5:
            continue
        shift = (Tr[mtr].median() - gmed).fillna(0.0)
        Tr.loc[mtr] = Tr.loc[mtr] - shift
        mte = bte == b
        if mte.any():
            Te.loc[mte] = Te.loc[mte] - shift

    fill = Tr.min() - 1.0                                   # half-minimum on the log2 scale
    Xtr = Tr.fillna(fill).values
    Xte = Te.fillna(fill).values

    # rank-based inverse-normal transform (train ECDF), then unit-variance scaling
    Rtr, Rte = _rank_int(Xtr, Xte)
    sc = StandardScaler().fit(Rtr)
    Ztr = np.clip(sc.transform(Rtr), -CLIP, CLIP)
    Zte = np.clip(sc.transform(Rte), -CLIP, CLIP)

    # prior-biology weighting: larger scale == weaker L2 penalty on these coefficients
    is_prior = np.array([_is_prior(c) for c in keep])
    w = np.where(is_prior, PRIOR_W, 1.0)

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr * w, y_train)
    p_full = clf.predict_proba(Zte * w)[:, 1]

    # compact knowledge-driven model on the a-priori AD biomarker panel only
    if is_prior.sum() < MIN_PANEL:
        return p_full
    pan = LogisticRegression(C=PANEL_C, max_iter=5000)
    pan.fit(Ztr[:, is_prior], y_train)
    p_pan = pan.predict_proba(Zte[:, is_prior])[:, 1]

    return (1.0 - PANEL_MIX) * p_full + PANEL_MIX * p_pan
"""log2 + per-sample median centring on a fixed core of near-completely quantified
proteins (robust to abundance-dependent missingness) + half-minimum imputation +
train-fold removal of site offsets, erythrocyte (blood-contamination) effects and
blood-CSF-barrier leakage (plasma-derived protein) effects, all estimated jointly
with diagnosis;
average of (a) L2 logistic on all well-quantified proteins plus detection indicators
of partially quantified (MNAR) proteins and
(b) L2 logistic on a prior-knowledge AD CSF biomarker panel."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Proteins reported as altered in AD CSF (glycolysis, 14-3-3, SMOC1, SPP1, chitinases,
# glial/neuronal injury, synaptic, complement/lipid transport, amyloid/tau-related).
PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "PKM", "ALDOA", "ALDOC",
    "GAPDH", "ENO1", "ENO2", "LDHB", "LDHA", "MDH1", "PGK1", "TPI1", "SPP1",
    "CHI3L1", "CHI3L2", "CHIT1", "GFAP", "NEFL", "NEFM", "NPTX2", "NPTXR", "VGF",
    "NRGN", "SNAP25", "GDA", "TMSB10", "PPIA", "PEBP1", "CLU", "APOE", "C4A", "C4B",
    "C3", "CFH", "C1QA", "C1QB", "C1QC", "MAPT", "APP", "CHGA", "SCG2", "TREM2",
    "CD44", "SOD1", "FABP3", "SYT1", "CALM1", "CALM2", "CALM3", "BASP1", "UCHL1",
}

# Erythrocyte-specific proteins: markers of blood contamination during lumbar puncture.
RBC = {"HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "CA1", "CA2", "PRDX2", "CAT",
       "SLC4A1", "SPTA1", "SPTB", "ANK1", "BLVRB", "HBQ1"}
RBC_COL = "__RBC_SCORE__"

# Predominantly liver/plasma-derived proteins whose CSF level reflects blood-CSF
# barrier permeability (albumin-quotient-like), not local brain production.
BBB = {"ALB", "HP", "APOA1", "APOA2", "APOB", "FGA", "FGB", "FGG", "AHSG", "HRG",
       "APOH", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHM", "IGKC", "ORM1",
       "ORM2", "A2M", "F2", "PLG", "KNG1", "C4BPA", "ITIH1", "ITIH2", "SERPINC1",
       "AMBP", "APOC3", "LRG1", "AFM"}
BBB_COL = "__BBB_SCORE__"
NUIS_COLS = [RBC_COL, BBB_COL]


def _genes(col):
    g = str(col).split("|")[0]
    return set(x.strip() for x in g.split(";") if x.strip())


def _score(F, det, genes, min_det):
    cols = [c for c in F.columns if (_genes(c) & genes) and det[c] >= min_det]
    if len(cols) < 1:
        return None
    R = F[cols].fillna(F[cols].min() - 1.0)
    R = (R - R.median()) / (R.std(ddof=0).replace(0, 1.0))
    return R.mean(axis=1).values


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # Label-free core set: proteins quantified in (nearly) every sample, so the per-sample
    # median is not shifted by which low-abundance proteins happened to be detected.
    det = L.notna().mean()
    core = det.index[det >= 0.95]
    if len(core) < 50:
        core = det.sort_values(ascending=False).index[:max(50, min(200, L.shape[1]))]
    ref = L[core].median(axis=1)
    ref = ref.fillna(L.median(axis=1))
    F = L.sub(ref, axis=0)

    # Label-free blood-contamination score: mean of centred erythrocyte protein levels
    # (missing -> half-minimum, since non-detection means low contamination).
    r = _score(F, det, RBC, 0.3)
    # Label-free barrier-leakage score: mean of centred plasma-derived protein levels.
    b = _score(F, det, BBB, 0.8)
    if r is not None:
        F[RBC_COL] = r
    if b is not None:
        F[BBB_COL] = b
    return F


def _sites(meta, n):
    try:
        if meta is None or "site" not in meta:
            return None
        s = np.array([str(v) for v in list(meta["site"])], dtype=object)
        return s if len(s) == n else None
    except Exception:
        return None


def _adjust(Xtr, Xte, y, s_tr, s_te, N_tr, N_te):
    """Remove per-site offsets and nuisance-score slopes (blood contamination, barrier
    leakage) estimated from X ~ site + diagnosis + nuisance on the training fold."""
    use_site = s_tr is not None and s_te is not None and len(set(s_tr.tolist())) >= 2
    if use_site:
        sites = np.array(sorted(set(s_tr.tolist())), dtype=object)
        Otr = np.stack([(s_tr == s).astype(float) for s in sites], axis=1)
        Ote = np.stack([(s_te == s).astype(float) for s in sites], axis=1)
    else:
        Otr = np.ones((len(Xtr), 1))
        Ote = np.ones((len(Xte), 1))
    cols = [Otr, y.astype(float)[:, None]]
    use_n = N_tr is not None and N_tr.shape[1] > 0
    if use_n:
        mu = N_tr.mean(axis=0, keepdims=True)
        ntr, nte = N_tr - mu, N_te - mu
        cols.append(ntr)
    if not use_site and not use_n:
        return Xtr, Xte
    D = np.column_stack(cols)
    B, *_ = np.linalg.lstsq(D, Xtr, rcond=None)
    k = Otr.shape[1]
    Atr, Ate = Xtr.copy(), Xte.copy()
    if use_site:
        off = B[:k]
        w = Otr.mean(axis=0)
        off = off - (w[:, None] * off).sum(axis=0, keepdims=True)  # keep overall level
        Atr = Atr - Otr @ off
        Ate = Ate - Ote @ off                                  # unseen site -> no shift
    if use_n:
        b = B[k + 1:]
        Atr = Atr - ntr @ b
        Ate = Ate - nte @ b
    return Atr, Ate


def _fit_lr(Xtr, y, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    ncols = [c for c in NUIS_COLS if c in F_train.columns]
    ncols = [c for c in ncols if np.std(F_train[c].values.astype(float)) > 1e-8]
    N_tr = F_train[ncols].values.astype(float) if ncols else None
    N_te = F_test[ncols].values.astype(float) if ncols else None
    drop = [c for c in NUIS_COLS if c in F_train.columns]
    F_train = F_train.drop(columns=drop)
    F_test = F_test.drop(columns=drop)
    cov = F_train.notna().mean()
    fill_all = F_train.min() - 1.0                          # half-minimum on the log2 scale
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))

    def prep(cols):
        Xtr = F_train[cols].fillna(fill_all[cols]).values
        Xte = F_test[cols].fillna(fill_all[cols]).values
        return _adjust(Xtr, Xte, y, s_tr, s_te, N_tr, N_te)

    # (a) full model: well-quantified protein levels + detection indicators of
    # partially quantified proteins (abundance-dependent missingness carries signal)
    keep = list(F_train.columns[cov >= 0.7])
    Xtr, Xte = prep(keep)
    low = list(F_train.columns[(cov >= 0.2) & (cov < 0.7)])
    if len(low) > 0:
        Itr = F_train[low].notna().values.astype(float)
        Ite = F_test[low].notna().values.astype(float)
        Itr, Ite = _adjust(Itr, Ite, y, s_tr, s_te, N_tr, N_te)
        Xtr = np.hstack([Xtr, Itr])
        Xte = np.hstack([Xte, Ite])
    p_full = _fit_lr(Xtr, y, Xte, C=0.05)

    # (b) prior biomarker panel model
    panel = [c for c in F_train.columns if (_genes(c) & PANEL) and cov[c] >= 0.5]
    if len(panel) >= 3:
        Ptr, Pte = prep(panel)
        p_panel = _fit_lr(Ptr, y, Pte, C=0.1)
        return 0.5 * p_full + 0.5 * p_panel
    return p_full
"""log2 + per-sample median centring + half-minimum imputation + per-batch protein centring
+ regression-out of erythrocyte / platelet contamination indices (fixed marker panels)
+ winsorised (±3 SD) standardised intensities + batch-centred missingness indicators for
mid-detection proteins + pathway module scores (mean z of AD-related pathway members)
+ L2 logistic regression with a weaker shrinkage prior on known AD-related plasma proteins,
ensembled with a low-dimensional L2 logistic regression restricted to the prior-biology
subspace (prior proteins + module scores); decision scores standardised on train and averaged."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# prior biology, grouped by pathway / biological module
MODULES = {
    "lipid": {"APOE", "CLU", "APOA1", "APOA4", "APOB", "APOC1", "APOC3", "APOD", "APOH", "APOM",
              "LCAT", "PLTP"},
    "complement": {"C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "C4BPA", "C5", "C7",
                   "C8A", "C9", "CFH", "CFB", "CFI", "CFD", "CLU", "SERPING1"},
    "neuroglial": {"GFAP", "NEFL", "NEFM", "YWHAG", "YWHAZ", "SMOC1", "SPP1", "CHI3L1", "CHIT1",
                   "NPTX2", "NPTXR", "VGF", "NRGN", "SNAP25", "GAP43", "MAPT", "APP", "BACE1",
                   "PPP3R1", "TREM2", "SPARCL1"},
    "inflammation": {"A2M", "CP", "TTR", "SERPINA3", "SERPINA1", "PPBP", "PF4", "IGFBP2", "VCAM1",
                     "CRP", "HP", "FGA", "FGB", "FGG", "VTN", "AMBP", "ORM1", "AHSG", "PRDX1",
                     "GSN", "PON1", "RBP4"},
}
AD_GENES = set().union(*MODULES.values())
# pre-analytical contamination panels (erythrocyte lysis, platelet activation); kept disjoint
# from the AD prior so prior proteins (e.g. PF4/PPBP) do not define the nuisance axes
CONTAM = {
    "erythrocyte": {"HBA1", "HBA2", "HBB", "HBD", "CA1", "CA2", "PRDX2", "CAT", "BLVRB", "SLC4A1",
                    "ALAD", "SOD1", "PRDX6", "GYPA", "SPTA1", "SPTB", "ANK1", "EPB42"},
    "platelet": {"TLN1", "ITGA2B", "ITGB3", "GP1BA", "GP9", "GP5", "MYH9", "FLNA", "VCL", "TPM4",
                 "ACTN1", "CFL1", "PFN1", "THBS1", "TAGLN2", "SELP", "VWF", "MMRN1", "F13A1"},
}
PRIOR_SCALE = 2.0
MODULE_SCALE = 2.0
CLIP = 3.0
PRIOR_MODEL_C = 0.1
PRIOR_MODEL_WEIGHT = 0.5


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame):
        if "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return np.zeros(n, dtype=object)
    if isinstance(meta, dict) and "batch" in meta:
        return np.asarray(meta["batch"]).astype(str)
    try:
        return np.asarray([m.get("batch", 0) if isinstance(m, dict) else m for m in meta]).astype(str)
    except Exception:
        return np.zeros(n, dtype=object)


def _batch_centre(Xtr, Xte, btr, bte, stat):
    """Remove per-batch offsets (stats from the training fold; unseen batch uses own label-free stats)."""
    g = stat(Xtr, axis=0)
    Xtr_c, Xte_c = Xtr.copy(), Xte.copy()
    for b in np.unique(np.concatenate([btr, bte])):
        mtr, mte = btr == b, bte == b
        if mtr.sum() >= 5:
            off = stat(Xtr[mtr], axis=0) - g
        elif mte.sum() >= 5:
            off = stat(Xte[mte], axis=0) - g
        else:
            off = 0.0
        Xtr_c[mtr] -= off
        Xte_c[mte] -= off
    return Xtr_c, Xte_c


def _genes(c):
    return [g.strip().upper() for g in str(c).split("|")[0].replace(",", ";").split(";")]


def _is_prior(cols):
    return np.array([any(g in AD_GENES for g in _genes(c)) for c in cols], dtype=bool)


def _prior_weights(cols):
    return np.where(_is_prior(cols), PRIOR_SCALE, 1.0)


def _remove_contamination(Str, Ste, cols):
    """Regress each standardised protein on per-sample contamination indices (train-fold OLS)."""
    gl = [_genes(c) for c in cols]
    Ttr, Tte = [], []
    for genes in CONTAM.values():
        idx = [i for i, g in enumerate(gl) if any(x in genes for x in g)]
        if len(idx) >= 3:
            Ttr.append(Str[:, idx].mean(axis=1))
            Tte.append(Ste[:, idx].mean(axis=1))
    if not Ttr:
        return Str, Ste
    Ttr = np.column_stack([np.ones(len(Str))] + Ttr)
    Tte = np.column_stack([np.ones(len(Ste))] + Tte)
    B, *_ = np.linalg.lstsq(Ttr, Str, rcond=None)
    return Str - Ttr @ B, Ste - Tte @ B


def _module_scores(Ztr, Zte, cols):
    """Mean standardised intensity of each pathway's measured members (label-free aggregation)."""
    gl = [_genes(c) for c in cols]
    Str, Ste = [], []
    for genes in MODULES.values():
        idx = [i for i, g in enumerate(gl) if any(x in genes for x in g)]
        if len(idx) >= 2:
            Str.append(Ztr[:, idx].mean(axis=1))
            Ste.append(Zte[:, idx].mean(axis=1))
    if not Str:
        return None, None
    Str, Ste = np.column_stack(Str), np.column_stack(Ste)
    sc = StandardScaler().fit(Str)
    return sc.transform(Str), sc.transform(Ste)


def _std_decision(clf, Ztr, Zte):
    """Decision scores on test, scaled by the spread of the training-fold scores."""
    dtr = clf.decision_function(Ztr)
    s = dtr.std()
    return (clf.decision_function(Zte) - dtr.mean()) / (s if s > 1e-12 else 1.0)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    pres = F_train.notna().mean()
    keep = F_train.columns[pres >= 0.7]                     # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    Xtr_c, Xte_c = _batch_centre(Xtr, Xte, btr, bte, np.median)
    sc0 = StandardScaler().fit(Xtr_c)
    # remove erythrocyte-lysis / platelet-activation axes (pre-analytical, not disease)
    Rtr, Rte = _remove_contamination(sc0.transform(Xtr_c), sc0.transform(Xte_c), keep)
    sc = StandardScaler().fit(Rtr)
    # winsorise: limit leverage of contaminated / outlying samples on any single protein
    Ctr = np.clip(sc.transform(Rtr), -CLIP, CLIP)
    Cte = np.clip(sc.transform(Rte), -CLIP, CLIP)
    wk = _prior_weights(keep)
    Ztr, Zte = Ctr * wk, Cte * wk

    # pathway module scores: coordinated shifts across many correlated proteins, less noisy
    Mod_tr, Mod_te = _module_scores(Ctr, Cte, keep)
    if Mod_tr is not None:
        Ztr = np.hstack([Ztr, MODULE_SCALE * Mod_tr])
        Zte = np.hstack([Zte, MODULE_SCALE * Mod_te])

    # detection indicators for proteins near the limit of detection (abundance-dependent missingness)
    mid = F_train.columns[(pres >= 0.2) & (pres < 0.7)]
    if len(mid) > 0:
        Mtr = F_train[mid].notna().values.astype(float)
        Mte = F_test[mid].notna().values.astype(float)
        Mtr_c, Mte_c = _batch_centre(Mtr, Mte, btr, bte, np.mean)
        msc = StandardScaler().fit(Mtr_c)
        wm = 0.5 * _prior_weights(mid)
        Ztr = np.hstack([Ztr, wm * msc.transform(Mtr_c)])
        Zte = np.hstack([Zte, wm * msc.transform(Mte_c)])

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    d_full = _std_decision(clf, Ztr, Zte)

    # low-dimensional model restricted to the prior-biology subspace (prior proteins + modules)
    pmask = _is_prior(keep)
    if pmask.sum() < 3:
        return d_full
    Ptr, Pte = Ctr[:, pmask], Cte[:, pmask]
    if Mod_tr is not None:
        Ptr = np.hstack([Ptr, Mod_tr])
        Pte = np.hstack([Pte, Mod_te])
    clf_p = LogisticRegression(C=PRIOR_MODEL_C, max_iter=5000)
    clf_p.fit(Ptr, y_train)
    d_prior = _std_decision(clf_p, Ptr, Pte)

    return (1.0 - PRIOR_MODEL_WEIGHT) * d_full + PRIOR_MODEL_WEIGHT * d_prior
"""log2 + per-sample median centring + half-minimum imputation + per-batch protein centring
+ winsorised (±3 SD) standardised intensities + batch-centred missingness indicators for
mid-detection proteins + L2 logistic regression with a weaker shrinkage prior (feature
up-weighting) on known AD-related plasma proteins."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# prior biology: proteins repeatedly implicated in AD / amyloid status in plasma & CSF proteomics
AD_GENES = {
    # apolipoproteins / lipid transport
    "APOE", "CLU", "APOA1", "APOA4", "APOB", "APOC1", "APOC3", "APOD", "APOH", "APOM", "LCAT", "PLTP",
    # complement cascade
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "C4BPA", "C5", "C7", "C8A", "C9",
    "CFH", "CFB", "CFI", "CFD", "CLU", "SERPING1",
    # glial / neuronal / synaptic injury
    "GFAP", "NEFL", "NEFM", "YWHAG", "YWHAZ", "SMOC1", "SPP1", "CHI3L1", "CHIT1", "NPTX2", "NPTXR",
    "VGF", "NRGN", "SNAP25", "GAP43", "MAPT", "APP", "BACE1", "PPP3R1", "TREM2", "SPARCL1",
    # acute-phase / inflammation / plasma AD signature proteins
    "A2M", "CP", "TTR", "SERPINA3", "SERPINA1", "PPBP", "PF4", "IGFBP2", "VCAM1", "CRP", "HP",
    "FGA", "FGB", "FGG", "VTN", "AMBP", "ORM1", "AHSG", "PRDX1", "GSN", "PON1", "RBP4",
}
PRIOR_SCALE = 2.0
CLIP = 3.0


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


def _prior_weights(cols):
    w = np.ones(len(cols))
    for i, c in enumerate(cols):
        genes = str(c).split("|")[0].replace(",", ";").split(";")
        if any(g.strip().upper() in AD_GENES for g in genes):
            w[i] = PRIOR_SCALE
    return w


def fit_predict(F_train, y_train, F_test, ctx):
    pres = F_train.notna().mean()
    keep = F_train.columns[pres >= 0.7]                     # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values

    btr = _batches(ctx.get("meta_train"), len(Xtr))
    bte = _batches(ctx.get("meta_test"), len(Xte))
    Xtr_c, Xte_c = _batch_centre(Xtr, Xte, btr, bte, np.median)
    sc = StandardScaler().fit(Xtr_c)
    wk = _prior_weights(keep)
    # winsorise: limit leverage of contaminated / outlying samples on any single protein
    Ztr = np.clip(sc.transform(Xtr_c), -CLIP, CLIP) * wk
    Zte = np.clip(sc.transform(Xte_c), -CLIP, CLIP) * wk

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
    return clf.predict_proba(Zte)[:, 1]
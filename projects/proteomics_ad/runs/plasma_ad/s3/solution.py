"""log2 + per-sample median centring + class-balanced per-batch protein centring (train-fold,
ComBat-with-design style: batch offset = mean of within-class batch means) +
half-minimum imputation + regression of erythrocyte/platelet contamination scores and
per-sample detection depth (train-fold coefficients) + z-score winsorization (+-3) +
prior-weighted (known AD biomarkers up-weighted, i.e. weaker L2 penalty) L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Plasma sample-quality contamination panels (Geyer et al. 2019, EMBO Mol Med)
_PANELS = {
    "erythrocyte": {"HBA1", "HBA2", "HBB", "HBD", "CA1", "CA2", "CAT", "PRDX2", "BLVRB",
                    "SLC4A1", "PRDX1", "PRDX6", "SOD1", "GAPDH", "BPGM", "SPTA1", "SPTB"},
    "platelet": {"PF4", "PPBP", "TLN1", "MYH9", "FLNA", "TPM4", "ITGA2B", "GP1BA", "VCL",
                 "CFL1", "PFN1", "TAGLN2", "THBS1", "ACTN1", "ZYX", "GP9", "ITGB3"},
}

# Prior AD biology (plasma/CSF proteomics literature): lipid transport, complement,
# acute-phase / inflammation, amyloid/tau-related and brain-leakage proteins
_AD_PRIOR = {
    "APOE", "CLU", "APOA1", "APOA2", "APOA4", "APOB", "APOC1", "APOC3", "APOD", "APOH", "APOM",
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "C4BPA", "CFH", "CFB", "CFI",
    "C5", "C9", "A2M", "SERPINA3", "CHI3L1", "HP", "CP", "SAA1", "SAA2", "CRP", "ORM1",
    "TTR", "AFM", "GSN", "PON1", "VTN", "ITIH4", "IGFBP2", "AHSG", "PLG", "BCHE",
    "GFAP", "NEFL", "YWHAG", "SMOC1", "SPP1", "NPTX2", "VGF", "APP", "MAPT", "PPY",
}
_PRIOR_SCALE = 2.0


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(ctx, key, n):
    meta = ctx.get(key)
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            b = np.asarray(meta["batch"].values)
        else:
            b = np.asarray(pd.DataFrame(meta)["batch"].values)
    except Exception:
        return None
    return b if len(b) == n else None


def _batch_centre(Ftr, Fte, ctx, y):
    btr = _batches(ctx, "meta_train", len(Ftr))
    bte = _batches(ctx, "meta_test", len(Fte))
    if btr is None or bte is None:
        return Ftr, Fte
    y = np.asarray(y)
    classes = np.unique(y)
    Ftr, Fte = Ftr.copy(), Fte.copy()
    gmean = Ftr.mean()                                      # overall train mean (observed values)
    means = {}
    for b in np.unique(btr):
        m = btr == b
        plain = Ftr.loc[m].mean()
        cms = []
        for c in classes:                                   # within-class batch means (train labels only)
            mc = m & (y == c)
            if mc.sum() >= 2:
                cms.append(Ftr.loc[mc].mean())
        if len(cms) == len(classes):
            cm = pd.concat(cms, axis=1)
            bal = cm.mean(axis=1)
            mb = bal.where(cm.notna().all(axis=1), plain)   # fall back where a class cell is empty
        else:
            mb = plain
        means[b] = mb.fillna(gmean)
        Ftr.loc[m] = Ftr.loc[m] - means[b] + gmean
    for b in np.unique(bte):
        m = bte == b
        if b in means:
            mb = means[b]
        else:                                               # unseen batch: label-free test mean
            mb = Fte.loc[m].mean().fillna(gmean)
        Fte.loc[m] = Fte.loc[m] - mb + gmean
    return Ftr, Fte


def _genes(col):
    return set(g.strip() for g in str(col).split("|")[0].split(";") if g.strip())


def _contamination_scores(Ztr, Zte, cols):
    """Mean clipped z-score of each contamination panel's quantified proteins."""
    Str, Ste = [], []
    for genes in _PANELS.values():
        idx = [i for i, c in enumerate(cols) if _genes(c) & genes]
        if len(idx) < 3:
            continue
        Str.append(np.clip(Ztr[:, idx], -3, 3).mean(axis=1))
        Ste.append(np.clip(Zte[:, idx], -3, 3).mean(axis=1))
    return Str, Ste


def fit_predict(F_train, y_train, F_test, ctx):
    # per-sample detection depth over ALL protein groups (label-free technical covariate)
    dtr = F_train.notna().mean(axis=1).values.astype(float)
    dte = F_test.notna().mean(axis=1).values.astype(float)
    dmu, dsd = dtr.mean(), dtr.std() + 1e-9
    dtr, dte = np.clip((dtr - dmu) / dsd, -3, 3), np.clip((dte - dmu) / dsd, -3, 3)

    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = _batch_centre(F_train[keep], F_test[keep], ctx, y_train)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    Str, Ste = _contamination_scores(Ztr, Zte, list(keep))
    Str.append(dtr)
    Ste.append(dte)
    Str, Ste = np.column_stack(Str), np.column_stack(Ste)
    mu = Str.mean(axis=0)                                   # regress out technical factors (train coefs)
    Atr = np.column_stack([np.ones(len(Str)), Str - mu])
    Ate = np.column_stack([np.ones(len(Ste)), Ste - mu])
    B, *_ = np.linalg.lstsq(Atr, Ztr, rcond=None)
    Ztr = Ztr - Atr[:, 1:] @ B[1:]
    Zte = Zte - Ate[:, 1:] @ B[1:]
    sc2 = StandardScaler().fit(Ztr)
    Ztr, Zte = sc2.transform(Ztr), sc2.transform(Zte)

    Ztr = np.clip(Ztr, -3.0, 3.0)                           # winsorize outlier measurements
    Zte = np.clip(Zte, -3.0, 3.0)

    # biology prior: scaling a feature by s shrinks its L2 penalty by s^2
    w = np.array([_PRIOR_SCALE if _genes(c) & _AD_PRIOR else 1.0 for c in keep])
    Ztr, Zte = Ztr * w, Zte * w

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    return clf.predict_proba(Zte)[:, 1]
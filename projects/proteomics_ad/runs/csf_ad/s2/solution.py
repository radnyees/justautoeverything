"""log2 + per-sample median centring on a fixed set of (nearly) always-quantified reference
proteins + half-minimum imputation + removal of erythrocyte, platelet (blood-contamination) and
plasma-leakage (blood-CSF barrier) signal + winsorization of standardized values at +-3 SD +
L2 logistic regression, blended with a fixed-sign prior-biology CSF AD panel score."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Literature CSF AD markers (Bader et al. 2020 and others): +1 = up in AD, -1 = down in AD
PANEL = {
    "SMOC1": 1, "YWHAG": 1, "PKM": 1, "ALDOA": 1, "MDH1": 1, "ENO1": 1, "GAPDH": 1,
    "CHI3L1": 1, "SPP1": 1, "NEFL": 1, "GFAP": 1,
    "NPTX2": -1, "VGF": -1, "NPTX1": -1,
}

# Erythrocyte contamination quality-marker panel (Geyer et al. 2019): blood from a traumatic
# tap raises these and also glycolytic enzymes / plasma proteins, confounding CSF AD signals
RBC = {
    "HBA1", "HBA2", "HBB", "HBD", "HBE1", "HBG1", "HBG2", "HBZ", "HBQ1", "HBM",
    "CA1", "CA2", "CAT", "PRDX2", "BLVRB", "SLC4A1", "SPTA1", "SPTB", "ANK1", "EPB42",
    "GYPA", "ALAD",
}

# Platelet contamination quality-marker panel (Geyer et al. 2019): platelet proteins enter CSF
# via traumatic taps / in-vitro clotting, a pre-analytical axis distinct from erythrocyte lysis
PLATELET = {
    "PF4", "PF4V1", "PPBP", "ITGA2B", "ITGB3", "GP1BA", "GP1BB", "GP5", "GP6", "GP9",
    "TLN1", "MMRN1", "TREML1", "CLEC1B", "F13A1", "MYL9", "TPM4", "TUBB1", "SELP",
    "THBS1", "VWF", "PDLIM1", "LIMS1",
}

# Strictly blood-derived proteins (liver / B-cell synthesis, high plasma:CSF ratio): their CSF
# level tracks blood-CSF barrier permeability (Q-albumin), which varies with age and non-AD
# neurological disease rather than AD pathology
PLASMA = {
    "ALB", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHM", "IGKC",
    "IGLC1", "IGLC2", "IGLC3", "A2M", "HP", "HPX", "FGA", "FGB", "FGG", "APOB",
    "SERPINA1", "ORM1", "ORM2", "AHSG", "FETUB", "HRG", "KNG1", "ITIH1", "ITIH2", "APOH",
}

CLIP = 3.0  # winsorize standardized protein values at +-3 SD (robust to spikes / imputed tails)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # reference set: proteins quantified in (nearly) every sample (label-free)
    frac = L.notna().mean(axis=0)
    ref = L.columns[frac >= 1.0]
    if len(ref) < 100:
        ref = L.columns[frac >= 0.95]
    if len(ref) < 50:
        ref = L.columns[frac >= 0.8]
    norm = L[ref].median(axis=1)
    return L.sub(norm, axis=0)


def _genes(col):
    g = str(col).split("|")[0]
    return [t.strip().upper() for t in g.split(";") if t.strip()]


def _remove_axis(Ztr, Zte, keep, genes):
    """Score = mean of standardized marker proteins; drop the markers and regress the score
    out of all remaining proteins (slopes fit on train only)."""
    idx = [j for j, c in enumerate(keep) if any(g in genes for g in _genes(c))]
    if len(idx) < 2:
        return Ztr, Zte, keep
    c_tr = Ztr[:, idx].mean(axis=1)
    c_te = Zte[:, idx].mean(axis=1)
    mu, sd = c_tr.mean(), c_tr.std() + 1e-9
    c_tr = (c_tr - mu) / sd
    c_te = (c_te - mu) / sd
    beta = Ztr.T @ c_tr / (c_tr @ c_tr)
    Ztr = Ztr - np.outer(c_tr, beta)
    Zte = Zte - np.outer(c_te, beta)
    rest = np.setdiff1d(np.arange(len(keep)), idx)
    return Ztr[:, rest], Zte[:, rest], keep[rest]


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr_df = F_train[keep].fillna(fill)
    Xte_df = F_test[keep].fillna(fill)
    sc = StandardScaler().fit(Xtr_df.values)
    Ztr = sc.transform(Xtr_df.values)
    Zte = sc.transform(Xte_df.values)

    # pre-analytical / barrier confounders: erythrocytes, platelets, then plasma leakage
    Ztr, Zte, keep = _remove_axis(Ztr, Zte, keep, RBC)
    Ztr, Zte, keep = _remove_axis(Ztr, Zte, keep, PLATELET)
    Ztr, Zte, keep = _remove_axis(Ztr, Zte, keep, PLASMA)

    # re-standardize residuals on train, then winsorize to limit single-value leverage
    mu = Ztr.mean(axis=0)
    sd = Ztr.std(axis=0) + 1e-9
    Ztr = np.clip((Ztr - mu) / sd, -CLIP, CLIP)
    Zte = np.clip((Zte - mu) / sd, -CLIP, CLIP)

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    d_tr = clf.decision_function(Ztr)
    d_te = clf.decision_function(Zte)
    s_lr = (d_te - d_tr.mean()) / (d_tr.std() + 1e-9)

    # fixed-sign panel score: mean of signed (confounder-corrected) panel proteins
    idx, signs = [], []
    for j, c in enumerate(keep):
        hits = [PANEL[g] for g in _genes(c) if g in PANEL]
        if hits:
            idx.append(j)
            signs.append(hits[0])
    if len(idx) == 0:
        return s_lr
    signs = np.asarray(signs, dtype=float)
    p_tr = (Ztr[:, idx] * signs).mean(axis=1)
    p_te = (Zte[:, idx] * signs).mean(axis=1)
    s_panel = (p_te - p_tr.mean()) / (p_tr.std() + 1e-9)

    return 0.5 * s_lr + 0.5 * s_panel
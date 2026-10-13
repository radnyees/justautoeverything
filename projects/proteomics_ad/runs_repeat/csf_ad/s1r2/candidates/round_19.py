"""log2 + per-sample median centring + half-minimum imputation + joint regression of two
label-free nuisance indices (erythrocyte contamination and plasma-protein leakage across the
blood-CSF barrier; train-fold slopes) + hierarchically prior-informed L2 logistic regression
(core literature AD panel proteins get a 2x wider coefficient scale, second-tier AD candidate
proteins whose direction/effect size is inconsistent across studies get 1.5x; the
nuisance-panel proteins themselves are excluded from the LR because after index regression
they keep only re-inflated residual noise), rank-averaged with a prior-biology two-sided CSF
module score: standardized mean z of 'tau-associated, up-in-AD' proteins minus standardized
mean z of 'synaptic/neurosecretory, down-in-AD' proteins (incl. the granin dense-core-vesicle
family). Each side is scaled to unit train-fold SD (label-free) so that the shared
brain-derived protein-content factor cancels symmetrically in the difference.
Second-tier candidates are used ONLY by the LR (sign learned), never by the fixed-sign module.
The plasma-leakage index excludes immunoglobulins: CSF Ig is a mixture of barrier transfer and
intrathecal synthesis (IgG-index biology, frequent in neurological non-AD patients), so it is
not a pure albumin-quotient analogue; Igs stay available to the LR as ordinary features.
Ranks are computed WITHIN each test site (harmonization only) so that both components carry
equal weight inside every cohort, matching the within-site scoring."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Proteins repeatedly reported as increased in AD CSF (tau/neuronal injury, glycolysis, glia).
AD_UP_PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "PKM", "ALDOA", "GAPDH", "ENO1", "ENO2", "MDH1",
    "LDHB", "PGK1", "TPI1", "PPIA", "CHI3L1", "SPP1", "NEFL", "MAPT", "GFAP",
}

# Synaptic / neurosecretory proteins repeatedly reported as decreased in clinical AD CSF.
# Granins (CHGA, CHGB, SCG3, SCG5) and PCSK1N are co-packaged with VGF/SCG2 in dense-core
# vesicles and belong to the same CSF neurosecretory module.
AD_DOWN_PANEL = {
    "NPTX2", "NPTXR", "NPTX1", "VGF", "SCG2", "NRN1",
    "CHGA", "CHGB", "SCG3", "SCG5", "PCSK1N",
}

# Second-tier AD candidates: biologically implicated but with inconsistent direction or
# effect size across CSF studies (lipid transport, complement, microglia, synaptic injury,
# neuropeptides, APP family, extracellular matrix). Used only by the LR, sign learned.
AD_TIER2_PANEL = {
    "APOE", "CLU", "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "CFH",
    "TREM2", "CHIT1", "CD44", "ALDOC", "LDHA", "PGAM1", "FABP3", "VSNL1", "BASP1",
    "NRGN", "SNAP25", "GAP43", "SYT1", "SST", "PENK", "PDYN", "CPE",
    "APP", "APLP1", "APLP2", "SPON1", "NCAN", "BCAN", "CNTN1", "CNTN2", "NEGR1", "THY1",
    "MDK", "PTN",
}

# Erythrocyte-specific proteins (Geyer et al. 2019 contamination panel): blood contamination index.
ERY_PANEL = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "CAT",
    "PRDX2", "SLC4A1", "BLVRB", "BPGM", "SPTA1", "SPTB", "ANK1", "EPB42", "GYPA", "ALAD",
}

# Liver/plasma-derived proteins that enter CSF mainly by blood-CSF barrier transfer
# (albumin-quotient analogue): plasma-leakage index. Immunoglobulins are deliberately
# excluded (partly intrathecally synthesized).
PLASMA_PANEL = {
    "ALB", "A2M", "APOA1", "APOA2", "APOB", "APOC3", "HP", "HPX", "FGA", "FGB", "FGG",
    "AHSG", "HRG", "ITIH1", "ITIH2", "APOH", "PLG", "F2", "ORM1", "ORM2", "SERPINC1",
    "LPA", "AFM",
}

# Prior-informed penalty: panel columns are up-scaled so their L2 prior variance is
# PRIOR_SCALE**2 (core) or TIER2_SCALE**2 (second tier) times wider than the rest.
PRIOR_SCALE = 2.0
TIER2_SCALE = 1.5


def _genes(col):
    return set(g.strip().upper() for g in str(col).split("|")[0].split(";") if g.strip())


def _test_sites(ctx, n):
    meta = ctx.get("meta_test") if isinstance(ctx, dict) else None
    try:
        if meta is not None and "site" in meta:
            s = np.asarray(meta["site"]).astype(str)
            if len(s) == n:
                return s
    except Exception:
        pass
    return np.zeros(n, dtype=str)


def _site_rank(x, sites):
    out = np.zeros(len(x), dtype=float)
    for s in np.unique(sites):
        m = sites == s
        out[m] = rankdata(x[m]) / m.sum()
    return out


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)

    # Nuisance correction (erythrocyte contamination + plasma leakage), jointly regressed out.
    # Indices and slopes are estimated on the training fold only, no labels.
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Ctr, Cte = [], []
    nuis_idx = set()
    for panel in (ERY_PANEL, PLASMA_PANEL):
        idx = [i for i, c in enumerate(keep) if _genes(c) & panel]
        if len(idx) >= 2:
            nuis_idx.update(idx)
            Ctr.append(((Xtr[:, idx] - mu[idx]) / sd[idx]).mean(axis=1))
            Cte.append(((Xte[:, idx] - mu[idx]) / sd[idx]).mean(axis=1))
    if Ctr:
        Ctr = np.column_stack(Ctr)
        Cte = np.column_stack(Cte)
        cm = Ctr.mean(axis=0)
        Ctr = Ctr - cm
        Cte = Cte - cm
        B = np.linalg.solve(Ctr.T @ Ctr + 1e-6 * np.eye(Ctr.shape[1]), Ctr.T @ (Xtr - mu))
        Xtr = Xtr - Ctr @ B
        Xte = Xte - Cte @ B

    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    Zte = sc.transform(Xte)

    up_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_UP_PANEL]
    down_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_DOWN_PANEL]
    panel_set = set(up_idx) | set(down_idx)
    tier2_idx = [i for i, c in enumerate(keep)
                 if (_genes(c) & AD_TIER2_PANEL) and i not in panel_set and i not in nuis_idx]

    # Hierarchical prior-informed L2: weakest shrinkage on core literature AD panel proteins,
    # intermediate on second-tier candidates. Nuisance-panel proteins (index definers) are
    # excluded: after regression they carry only residual noise that standardization re-inflates.
    w = np.ones(Ztr.shape[1])
    w[tier2_idx] = TIER2_SCALE
    w[up_idx + down_idx] = PRIOR_SCALE
    lr_cols = np.array([i for i in range(Ztr.shape[1]) if i not in nuis_idx or i in panel_set])
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit((Ztr * w)[:, lr_cols], y_train)
    p_lr = clf.predict_proba((Zte * w)[:, lr_cols])[:, 1]

    # Prior-knowledge module: signs fixed a priori, no fitting to labels.
    # Each side is standardized by its train-fold SD so both sides weigh equally.
    if len(up_idx) < 3:
        return p_lr
    up_tr = Ztr[:, up_idx].mean(axis=1)
    module = Zte[:, up_idx].mean(axis=1) / (up_tr.std() + 1e-9)
    if len(down_idx) >= 2:
        down_tr = Ztr[:, down_idx].mean(axis=1)
        module = module - Zte[:, down_idx].mean(axis=1) / (down_tr.std() + 1e-9)

    sites = _test_sites(ctx, len(p_lr))
    return 0.5 * _site_rank(p_lr, sites) + 0.5 * _site_rank(module, sites)
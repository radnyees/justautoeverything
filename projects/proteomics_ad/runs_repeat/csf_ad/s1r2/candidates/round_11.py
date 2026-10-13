"""log2 + per-sample median centring + half-minimum imputation + joint regression of two
label-free nuisance indices (erythrocyte contamination and plasma-protein leakage across the
blood-CSF barrier; train-fold slopes) + L2 logistic regression, rank-averaged with a
prior-biology two-sided CSF module score built hierarchically: proteins are first averaged
within biological sub-modules (matrisome SMOC1/SPP1, 14-3-3, glycolysis, glia, neuronal
injury; pentraxin/synaptic, dense-core-vesicle granins), then sub-modules are averaged with
equal weight, so no single large correlated family (glycolytic enzymes) dominates.
Ranks are computed WITHIN each test site (harmonization only) so that both components carry
equal weight inside every cohort, matching the within-site scoring."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Proteins repeatedly reported as increased in AD CSF, grouped into biological sub-modules.
AD_UP_MODULES = [
    {"SMOC1", "SPP1"},                                                   # matrisome / tau-coupled
    {"YWHAG", "YWHAZ", "YWHAB"},                                         # 14-3-3
    {"PKM", "ALDOA", "GAPDH", "ENO1", "MDH1", "LDHB", "PGK1", "TPI1", "PPIA"},  # glycolysis
    {"CHI3L1", "GFAP"},                                                  # glial activation
    {"NEFL", "MAPT", "ENO2"},                                            # neuronal injury
]

# Synaptic / neurosecretory proteins repeatedly reported as decreased in clinical AD CSF.
AD_DOWN_MODULES = [
    {"NPTX2", "NPTXR", "NPTX1", "NRN1"},                                 # pentraxin / synaptic
    {"VGF", "SCG2", "CHGA", "CHGB", "SCG3", "SCG5", "PCSK1N"},           # dense-core vesicles
]

# Erythrocyte-specific proteins (Geyer et al. 2019 contamination panel): blood contamination index.
ERY_PANEL = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "CAT",
    "PRDX2", "SLC4A1", "BLVRB", "BPGM", "SPTA1", "SPTB", "ANK1", "EPB42", "GYPA", "ALAD",
}

# Liver/plasma-derived proteins that enter CSF mainly by blood-CSF barrier transfer
# (albumin-quotient analogue): plasma-leakage index.
PLASMA_PANEL = {
    "ALB", "A2M", "APOA1", "APOA2", "APOB", "APOC3", "HP", "HPX", "FGA", "FGB", "FGG",
    "AHSG", "HRG", "ITIH1", "ITIH2", "APOH", "PLG", "F2", "ORM1", "ORM2", "SERPINC1",
    "IGHM", "IGHA1", "IGHG1", "IGHG2", "IGKC", "LPA", "AFM",
}


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


def _module_score(Z, keep, modules):
    """Mean over sub-modules of the mean z within each sub-module; returns (score, n_proteins)."""
    subs, n_prot = [], 0
    for mod in modules:
        idx = [i for i, c in enumerate(keep) if _genes(c) & mod]
        if idx:
            subs.append(Z[:, idx].mean(axis=1))
            n_prot += len(idx)
    if not subs:
        return None, 0
    return np.column_stack(subs).mean(axis=1), n_prot


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
    for panel in (ERY_PANEL, PLASMA_PANEL):
        idx = [i for i, c in enumerate(keep) if _genes(c) & panel]
        if len(idx) >= 2:
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

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    p_lr = clf.predict_proba(Zte)[:, 1]

    # Prior-knowledge module: signs fixed a priori, hierarchical (sub-module-balanced) averaging.
    up, n_up = _module_score(Zte, keep, AD_UP_MODULES)
    if up is None or n_up < 3:
        return p_lr
    module = up
    down, n_down = _module_score(Zte, keep, AD_DOWN_MODULES)
    if down is not None and n_down >= 2:
        module = module - down

    sites = _test_sites(ctx, len(p_lr))
    return 0.5 * _site_rank(p_lr, sites) + 0.5 * _site_rank(module, sites)
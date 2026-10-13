"""log2 + per-sample median centring + half-minimum imputation + joint regression of two
label-free nuisance indices (erythrocyte contamination and plasma-protein leakage across the
blood-CSF barrier; train-fold slopes) + L2 logistic regression trained with site x class
balancing weights (each cohort contributes a class-balanced loss, so cohort identity carries no
label information and site-marker proteins cannot act as AD predictors), rank-averaged with a
prior-biology two-sided CSF module score:
mean z of 'tau-associated, up-in-AD' proteins minus mean z of 'synaptic/neurosecretory,
down-in-AD' proteins (incl. the granin dense-core-vesicle family).
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


def _sites(ctx, key, n):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
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


def _site_class_weights(y, sites):
    """Weights making every (site, class) cell contribute equally: label independent of site."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(sites):
        for c in np.unique(y):
            m = (sites == s) & (y == c)
            if m.sum() > 0:
                w[m] = 1.0 / m.sum()
    return w * len(w) / w.sum()


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

    y_arr = np.asarray(y_train)
    w = _site_class_weights(y_arr, _sites(ctx, "meta_train", len(y_arr)))
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_arr, sample_weight=w)
    p_lr = clf.predict_proba(Zte)[:, 1]

    # Prior-knowledge module: signs fixed a priori, no fitting.
    up_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_UP_PANEL]
    down_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_DOWN_PANEL]
    if len(up_idx) < 3:
        return p_lr
    module = Zte[:, up_idx].mean(axis=1)
    if len(down_idx) >= 2:
        module = module - Zte[:, down_idx].mean(axis=1)

    sites = _sites(ctx, "meta_test", len(p_lr))
    return 0.5 * _site_rank(p_lr, sites) + 0.5 * _site_rank(module, sites)
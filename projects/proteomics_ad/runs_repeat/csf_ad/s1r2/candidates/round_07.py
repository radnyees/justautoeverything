"""log2 + per-sample median centring + half-minimum imputation + erythrocyte-contamination
regression (label-free, train-fold slopes) + L2 logistic regression, rank-averaged with a
prior-biology two-sided CSF module score:
mean z of 'tau-associated, up-in-AD' proteins minus mean z of 'synaptic, down-in-AD' proteins.
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
AD_DOWN_PANEL = {"NPTX2", "NPTXR", "NPTX1", "VGF", "SCG2", "NRN1"}

# Erythrocyte-specific proteins (Geyer et al. 2019 contamination panel): blood contamination index.
ERY_PANEL = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "CAT",
    "PRDX2", "SLC4A1", "BLVRB", "BPGM", "SPTA1", "SPTB", "ANK1", "EPB42", "GYPA", "ALAD",
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


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values.astype(float)
    Xte = F_test[keep].fillna(fill).values.astype(float)

    # Blood-contamination correction: slopes estimated on the training fold only, no labels.
    ery_idx = [i for i, c in enumerate(keep) if _genes(c) & ERY_PANEL]
    if len(ery_idx) >= 2:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-9
        ctr = ((Xtr[:, ery_idx] - mu[ery_idx]) / sd[ery_idx]).mean(axis=1)
        cte = ((Xte[:, ery_idx] - mu[ery_idx]) / sd[ery_idx]).mean(axis=1)
        cm = ctr.mean()
        ctr = ctr - cm
        cte = cte - cm
        beta = (Xtr - mu).T @ ctr / (ctr @ ctr + 1e-9)
        Xtr = Xtr - np.outer(ctr, beta)
        Xte = Xte - np.outer(cte, beta)

    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    Zte = sc.transform(Xte)

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    p_lr = clf.predict_proba(Zte)[:, 1]

    # Prior-knowledge module: signs fixed a priori, no fitting.
    up_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_UP_PANEL]
    down_idx = [i for i, c in enumerate(keep) if _genes(c) & AD_DOWN_PANEL]
    if len(up_idx) < 3:
        return p_lr
    module = Zte[:, up_idx].mean(axis=1)
    if len(down_idx) >= 2:
        module = module - Zte[:, down_idx].mean(axis=1)

    sites = _test_sites(ctx, len(p_lr))
    return 0.5 * _site_rank(p_lr, sites) + 0.5 * _site_rank(module, sites)
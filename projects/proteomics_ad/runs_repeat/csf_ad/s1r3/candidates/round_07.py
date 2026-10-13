"""log2 + per-sample median centring on a near-completely-quantified reference protein set
+ label-free removal of erythrocyte (blood-contamination) signal
+ half-minimum imputation + L2 logistic regression, ensembled with a prior-biology panel model
(known CSF AD markers). Training samples are weighted so classes are balanced within each site
(removes site/prevalence confounding from the fitted models)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Known CSF AD-associated proteins (Bader 2020 tau-correlated set, glycolysis / 14-3-3 module,
# synaptic, glial / inflammatory, complement, lipid transport).
PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "PKM", "ALDOA", "ENO1", "ENO2",
    "GAPDH", "LDHA", "LDHB", "MDH1", "PGK1", "TPI1", "PGAM1", "PPIA", "SOD1", "PARK7",
    "TMSB10", "PEBP1", "CFL1", "CALM1", "CALM2", "CALM3", "CHI3L1", "CHI3L2", "SPP1",
    "GFAP", "NEFL", "NEFM", "NPTX1", "NPTX2", "NPTXR", "VGF", "NRGN", "SNAP25", "GAP43",
    "BASP1", "MAPT", "APOE", "CLU", "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH",
    "TREM2", "CD44", "ITGB2", "SCG2", "CHGA", "PDYN", "VSTM2A", "NCAN", "MIF",
}

# Erythrocyte-specific proteins: their CSF levels track blood contamination at lumbar puncture
# (Geyer / Bader sample-quality marker panel), not disease.
ERY = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "PRDX2", "CAT",
    "SLC4A1", "SPTA1", "SPTB", "ALAS2", "BLVRB", "EPB42", "ANK1", "GYPA", "HMBS", "SELENBP1",
}


def _gene_match(cols, genes_set):
    out = []
    for c in cols:
        genes = str(c).split("|")[0].split(";")
        if any(g.strip().upper() in genes_set for g in genes):
            out.append(c)
    return out


def _remove_contamination(L):
    det = L.notna().mean()
    ery = _gene_match(L.columns, ERY)
    ery_ok = [c for c in ery if det[c] >= 0.5]
    if len(ery_ok) < 3:
        return L
    E = L[ery_ok]
    Z = (E - E.mean()) / (E.std() + 1e-9)
    s = Z.mean(axis=1, skipna=True)
    s = s.fillna(s.min())
    s = (s - s.mean()).values

    X = L.drop(columns=ery).values
    M = ~np.isnan(X)
    n = M.sum(axis=0).astype(float)
    S = np.where(M, s[:, None], 0.0)
    sm = S.sum(axis=0) / np.maximum(n, 1)
    xm = np.nanmean(np.where(M, X, np.nan), axis=0)
    ds = np.where(M, s[:, None] - sm, 0.0)
    dx = np.where(M, X - xm, 0.0)
    var = (ds ** 2).sum(axis=0)
    cov = (ds * dx).sum(axis=0)
    slope = np.where((n >= 10) & (var > 1e-9), cov / np.maximum(var, 1e-9), 0.0)
    Xr = X - slope[None, :] * s[:, None]  # NaN stays NaN
    return pd.DataFrame(Xr, index=L.index, columns=L.columns.drop(ery))


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # Reference set: proteins quantified in (almost) every sample, so the per-sample median is
    # not biased by abundance-dependent missingness of low-abundance proteins.
    det = L.notna().mean()
    ref = L.columns[det >= 0.95]
    if len(ref) < 50:
        ref = det.sort_values(ascending=False).index[:max(50, len(det) // 5)]
    L = L.sub(L[ref].median(axis=1), axis=0)
    return _remove_contamination(L)


def _panel_cols(cols):
    return _gene_match(cols, PANEL)


def _site_balanced_weights(y, ctx):
    """Within each site, give both classes equal total weight (site keeps its size share)."""
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        site = np.asarray(pd.DataFrame(meta)["site"]).astype(str)
    except Exception:
        return None
    if len(site) != len(y):
        return None
    w = np.ones(len(y), dtype=float)
    for s in np.unique(site):
        m = site == s
        n_s = m.sum()
        classes = np.unique(y[m])
        if len(classes) < 2:
            continue
        for c in classes:
            mc = m & (y == c)
            w[mc] = n_s / (len(classes) * mc.sum())
    return w / w.mean()


def _fit_score(Xtr, y, Xte, C, w=None):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y, sample_weight=w)
    s_tr = clf.decision_function(sc.transform(Xtr))
    s_te = clf.decision_function(sc.transform(Xte))
    return (s_te - s_tr.mean()) / (s_tr.std() + 1e-9)


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    w = _site_balanced_weights(y, ctx)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr_df = F_train[keep].fillna(fill)
    Xte_df = F_test[keep].fillna(fill)

    s_full = _fit_score(Xtr_df.values, y, Xte_df.values, C=0.05, w=w)

    pcols = _panel_cols(keep)
    if len(pcols) >= 3:
        s_panel = _fit_score(Xtr_df[pcols].values, y, Xte_df[pcols].values, C=0.1, w=w)
        return 0.5 * s_full + 0.5 * s_panel
    return s_full
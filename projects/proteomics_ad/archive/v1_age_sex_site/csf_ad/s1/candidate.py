"""log2 + per-sample median centring on a fixed reference set of (near-)completely quantified
proteins + label-free regression of an erythrocyte (blood-contamination) score out of every
protein + half-minimum imputation; average of a full-proteome L2 logistic regression, an L2
logistic regression on a prior-biology CSF AD marker panel, and a fixed prior log-ratio module
score (tau/glycolysis/14-3-3/synaptic-leakage up-module minus neurosecretory down-module)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Known CSF AD-associated proteins (Bader et al. 2020 tau/glycolysis panel, 14-3-3, synaptic,
# glial/inflammatory, complement and lipid-transport markers).
PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "SPP1", "CHI3L1", "CHI3L2",
    "MDH1", "PKM", "ALDOA", "ALDOC", "GAPDH", "ENO1", "ENO2", "LDHB", "PGK1", "TPI1",
    "NEFL", "NEFM", "GFAP", "NPTX2", "NPTXR", "VGF", "NRGN", "SNAP25", "SYT1", "GAP43",
    "TREM2", "MAPT", "APP", "CLU", "APOE", "C3", "C4A", "C4B", "C1QA", "C1QB", "C1QC",
    "CFH", "SPARCL1", "ITM2B", "CD44", "SOD1", "PEBP1", "CALM1", "FABP3", "VIM", "CHIT1",
}

# Proteins reported increased in AD CSF: tau-correlated glycolysis / 14-3-3 / redox cluster
# (Bader 2020, Higginbotham 2020), glial activation, and synaptic-degeneration leakage markers
# (neurogranin, SNAP-25, GAP-43, synaptotagmin-1, VAMP2 are elevated in AD CSF).
UP_MODULE = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "SPP1", "CHI3L1", "MDH1", "PKM",
    "ALDOA", "ALDOC", "GAPDH", "ENO1", "ENO2", "LDHA", "LDHB", "PGK1", "PGAM1", "TPI1",
    "PPIA", "PARK7", "CD44", "MAPT",
    "NRGN", "SNAP25", "GAP43", "SYT1", "VAMP2",
}
# Synaptic / neurosecretory proteins reported decreased in AD CSF
DOWN_MODULE = {"NPTX2", "NPTXR", "VGF", "SCG2", "CHGB"}

# Erythrocyte-specific proteins: blood-contamination quality marker panel (Bader et al. 2020)
ERY = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "CA1", "CA2", "PRDX2",
    "CAT", "BLVRB", "SLC4A1", "SPTA1", "SPTB", "ALAS2", "EPB42", "ANK1",
}


def _gene_cols(columns, genes_set):
    out = []
    for c in columns:
        genes = str(c).split("|")[0].split(";")
        if any(g.strip().upper() in genes_set for g in genes):
            out.append(c)
    return out


def _remove_contamination(L):
    pres = L.notna().mean()
    ery = [c for c in _gene_cols(L.columns, ERY) if pres[c] >= 0.3]
    if len(ery) < 2:
        return L
    E = L[ery]
    E = E.fillna(E.min() - 1.0)                             # missing = low contamination
    sd = E.std().replace(0, 1.0).fillna(1.0)
    score = ((E - E.mean()) / sd).mean(axis=1).values
    X = L.values.copy()
    for j in range(X.shape[1]):
        if L.columns[j] in ery:
            continue
        x = X[:, j]
        m = ~np.isnan(x)
        if m.sum() < 10:
            continue
        s = score[m] - score[m].mean()
        v = (s * s).sum()
        if v <= 1e-12:
            continue
        beta = (s * (x[m] - x[m].mean())).sum() / v
        X[m, j] = x[m] - beta * s
    return pd.DataFrame(X, index=L.index, columns=L.columns)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.replace([np.inf, -np.inf], np.nan)
    pres = L.notna().mean()
    # Reference set: proteins quantified in (nearly) every sample, so the normalization factor
    # is not biased by abundance-dependent dropout of low-abundance proteins.
    ref = pres.index[pres >= 0.99]
    if len(ref) < 100:
        ref = pres.sort_values(ascending=False).index[:max(100, len(ref))]
    factor = L[ref].median(axis=1, skipna=True)
    factor = factor.fillna(L.median(axis=1))
    L = L.sub(factor, axis=0)
    # Blood contamination inflates erythrocyte-rich proteins (glycolytic enzymes, 14-3-3);
    # regress the erythrocyte score out of each protein (label-free).
    return _remove_contamination(L)


def _fit_lr(F_train, y_train, F_test, cols, C):
    fill = F_train[cols].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[cols].fillna(fill).values
    Xte = F_test[cols].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _module_mean(F, cols, mu, sd):
    if len(cols) == 0:
        return np.zeros(len(F))
    Z = (F[cols] - mu[cols]) / sd[cols]
    m = Z.mean(axis=1, skipna=True).values                  # average over quantified proteins
    return np.nan_to_num(m, nan=0.0)


def _module_score(F_train, y_train, F_test, pres):
    up = [c for c in _gene_cols(F_train.columns, UP_MODULE) if pres[c] >= 0.5]
    dn = [c for c in _gene_cols(F_train.columns, DOWN_MODULE) if pres[c] >= 0.5]
    if len(up) < 2 or len(dn) < 1:
        return None
    cols = up + dn
    mu = F_train[cols].mean()
    sd = F_train[cols].std().replace(0, 1.0).fillna(1.0)
    s_tr = _module_mean(F_train, up, mu, sd) - _module_mean(F_train, dn, mu, sd)
    s_te = _module_mean(F_test, up, mu, sd) - _module_mean(F_test, dn, mu, sd)
    # 1-D Platt calibration on train (sign is learned too; only 2 parameters)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(s_tr.reshape(-1, 1), y_train)
    return clf.predict_proba(s_te.reshape(-1, 1))[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    pres = F_train.notna().mean()
    keep = F_train.columns[pres >= 0.7]                     # quantified in >= 70% of train samples
    preds = [_fit_lr(F_train, y_train, F_test, keep, C=0.05)]

    panel = [c for c in _gene_cols(F_train.columns, PANEL) if pres[c] >= 0.5]
    if len(panel) >= 3:
        preds.append(_fit_lr(F_train, y_train, F_test, panel, C=0.1))

    p_mod = _module_score(F_train, y_train, F_test, pres)
    if p_mod is not None:
        preds.append(p_mod)
    return np.mean(preds, axis=0)
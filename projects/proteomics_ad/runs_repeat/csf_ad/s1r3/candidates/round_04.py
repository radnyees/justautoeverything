"""log2 + per-sample median centring on a near-completely-quantified reference protein set
+ half-minimum imputation + L2 logistic regression, ensembled with a prior-biology panel model
(known CSF AD markers)."""
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


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # Reference set: proteins quantified in (almost) every sample, so the per-sample median is
    # not biased by abundance-dependent missingness of low-abundance proteins.
    det = L.notna().mean()
    ref = L.columns[det >= 0.95]
    if len(ref) < 50:
        ref = det.sort_values(ascending=False).index[:max(50, len(det) // 5)]
    return L.sub(L[ref].median(axis=1), axis=0)


def _panel_cols(cols):
    out = []
    for c in cols:
        genes = str(c).split("|")[0].split(";")
        if any(g.strip().upper() in PANEL for g in genes):
            out.append(c)
    return out


def _fit_score(Xtr, y, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y)
    s_tr = clf.decision_function(sc.transform(Xtr))
    s_te = clf.decision_function(sc.transform(Xte))
    return (s_te - s_tr.mean()) / (s_tr.std() + 1e-9)


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr_df = F_train[keep].fillna(fill)
    Xte_df = F_test[keep].fillna(fill)

    s_full = _fit_score(Xtr_df.values, y, Xte_df.values, C=0.05)

    pcols = _panel_cols(keep)
    if len(pcols) >= 3:
        s_panel = _fit_score(Xtr_df[pcols].values, y, Xte_df[pcols].values, C=0.1)
        return 0.5 * s_full + 0.5 * s_panel
    return s_full
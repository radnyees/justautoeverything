"""log2 + per-sample median centring + half-minimum imputation + L2 logistic regression,
ensembled with a prior-biology panel model (known CSF AD markers). The panel uses a looser
detection filter so low-abundance markers (abundance-dependent missingness) are retained."""
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
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


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
    det = F_train.notna().mean()
    keep = F_train.columns[det >= 0.7]                      # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr_df = F_train[keep].fillna(fill)
    Xte_df = F_test[keep].fillna(fill)

    s_full = _fit_score(Xtr_df.values, y, Xte_df.values, C=0.05)

    # Panel: looser detection filter; missing low-abundance markers imputed at half-minimum
    pcols = _panel_cols(F_train.columns[det >= 0.3])
    if len(pcols) >= 3:
        pfill = F_train[pcols].min() - 1.0
        Ptr = F_train[pcols].fillna(pfill).values
        Pte = F_test[pcols].fillna(pfill).values
        s_panel = _fit_score(Ptr, y, Pte, C=0.1)
        return 0.5 * s_full + 0.5 * s_panel
    return s_full
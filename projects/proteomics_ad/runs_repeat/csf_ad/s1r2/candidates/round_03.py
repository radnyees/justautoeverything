"""log2 + per-sample median centring + half-minimum imputation + L2 logistic regression,
rank-averaged with a prior-biology two-sided CSF module score:
mean z of 'tau-associated, up-in-AD' proteins minus mean z of 'synaptic, down-in-AD' proteins."""
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


def _genes(col):
    return set(g.strip().upper() for g in str(col).split("|")[0].split(";") if g.strip())


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill)
    Xte = F_test[keep].fillna(fill)
    sc = StandardScaler().fit(Xtr.values)
    Ztr = sc.transform(Xtr.values)
    Zte = sc.transform(Xte.values)

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

    n = len(p_lr)
    return 0.5 * rankdata(p_lr) / n + 0.5 * rankdata(module) / n
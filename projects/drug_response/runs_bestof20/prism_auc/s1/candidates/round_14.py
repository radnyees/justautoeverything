"""Two-stage model: (1) per-drug shrunken lineage-mean AUC (NaN-aware, empirical-Bayes-style shrinkage
toward the drug's grand mean), (2) multi-output ridge on top-5,000 variable standardized genes fitted to
the residuals (missing residuals set to 0). Prediction = lineage effect + expression ridge."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SHRINK_K = 5.0  # pseudo-count of lines pulling a lineage mean toward the drug grand mean


def _lineage(meta, n):
    if meta is None:
        return np.array(["NA"] * n)
    if isinstance(meta, pd.DataFrame) and "OncotreeLineage" in meta.columns:
        return meta["OncotreeLineage"].astype(str).fillna("NA").values
    return np.array(["NA"] * n)


def fit_predict(F_train, y_train, F_test, ctx):
    lin_tr = _lineage(ctx.get("meta_train"), len(F_train))
    lin_te = _lineage(ctx.get("meta_test"), len(F_test))

    Yraw = y_train.values.astype(float)
    obs = ~np.isnan(Yraw)
    grand = np.nanmean(Yraw, axis=0)
    grand = np.where(np.isnan(grand), 0.0, grand)
    D = np.where(obs, Yraw - grand, 0.0)  # deviations, 0 where missing

    # shrunken per-lineage, per-drug effects
    effects = {}
    for L in np.unique(lin_tr):
        m = lin_tr == L
        s = D[m].sum(axis=0)
        c = obs[m].sum(axis=0)
        effects[L] = s / (c + SHRINK_K)
    zero = np.zeros(Yraw.shape[1])
    E_tr = np.vstack([effects.get(L, zero) for L in lin_tr])
    E_te = np.vstack([effects.get(L, zero) for L in lin_te])

    R = np.where(obs, Yraw - grand - E_tr, 0.0)

    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    model = Ridge(alpha=5000.0).fit(Xtr, R)
    return grand + E_te + model.predict(Xte)
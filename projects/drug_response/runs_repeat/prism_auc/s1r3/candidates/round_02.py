"""Lineage-shrunk per-drug means + multi-output ridge on top-5,000 variable genes fitted to
lineage-adjusted residual AUCs (missing residuals filled with 0 = the lineage-adjusted mean)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SHRINK_K = 5.0  # pseudo-count shrinking lineage means toward the global drug mean


def _lineages(meta, n):
    if meta is None:
        return np.array(["NA"] * n)
    if isinstance(meta, pd.DataFrame) and "OncotreeLineage" in meta.columns:
        return meta["OncotreeLineage"].astype(str).fillna("NA").values
    return np.array(["NA"] * n)


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)

    Yraw = y_train.values.astype(float)
    obs = ~np.isnan(Yraw)
    mu = np.nanmean(Yraw, axis=0)
    R = np.where(obs, Yraw - mu, 0.0)

    lin_tr = _lineages(ctx.get("meta_train"), len(F_train))
    lin_te = _lineages(ctx.get("meta_test"), len(F_test))

    # empirical-Bayes shrunk lineage effect per drug (observed values only)
    effects = {}
    for lin in np.unique(lin_tr):
        m = lin_tr == lin
        s = R[m].sum(axis=0)
        n = obs[m].sum(axis=0)
        effects[lin] = s / (n + SHRINK_K)
    zero = np.zeros(Yraw.shape[1])
    L_tr = np.vstack([effects.get(l, zero) for l in lin_tr])
    L_te = np.vstack([effects.get(l, zero) for l in lin_te])

    resid = np.where(obs, R - L_tr, 0.0)
    model = Ridge(alpha=5000.0).fit(Xtr, resid)
    return mu[None, :] + L_te + model.predict(Xte)
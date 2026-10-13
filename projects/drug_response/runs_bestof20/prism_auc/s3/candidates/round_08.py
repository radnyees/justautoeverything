"""Top-5,000 most variable genes, standardized, ridge regression for all drugs with a PER-DRUG
regularization strength chosen by grouped inner CV (SVD makes the whole alpha path cheap).
Missing training AUCs are filled with the drug's training mean when fitting."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.array([300., 1000., 2000., 5000., 1e4, 2e4, 5e4, 1e5, 3e5])


def _ridge_path(Xtr, Ytr, Xte, alphas):
    """Predictions (n_alpha, n_te, n_drugs) of intercept ridge for each alpha via SVD."""
    xm = Xtr.mean(0)
    ym = Ytr.mean(0)
    Xc = Xtr - xm
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ (Ytr - ym)                      # (k, d)
    XteV = (Xte - xm) @ Vt.T                    # (n_te, k)
    out = np.empty((len(alphas), Xte.shape[0], Ytr.shape[1]))
    for i, a in enumerate(alphas):
        d = s / (s ** 2 + a)
        out[i] = XteV @ (d[:, None] * UtY) + ym
    return out


def _pearson_cols(P, Y):
    """Per-column Pearson r between P (n, d) and Y (n, d) ignoring NaNs in Y."""
    M = ~np.isnan(Y)
    Mf = M.astype(float)
    n = np.maximum(Mf.sum(0), 1.0)
    Yf = np.where(M, Y, 0.0)
    mp = (P * Mf).sum(0) / n
    my = Yf.sum(0) / n
    dp = (P - mp) * Mf
    dy = (Yf - my) * Mf
    cov = (dp * dy).sum(0)
    den = np.sqrt((dp ** 2).sum(0) * (dy ** 2).sum(0)) + 1e-12
    return cov / den


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    n, d = Yraw.shape

    # inner grouped CV over the alpha path
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(n)
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros((len(ALPHAS), n, d))
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, groups=groups):
        Ysub = Yraw[tr]
        mu = np.nanmean(Ysub, axis=0)
        mu = np.where(np.isnan(mu), np.nanmean(Yraw), mu)
        Yfill = np.where(np.isnan(Ysub), mu, Ysub)
        oof[:, va, :] = _ridge_path(Xtr[tr], Yfill, Xtr[va], ALPHAS)
    scores = np.stack([_pearson_cols(oof[i], Yraw) for i in range(len(ALPHAS))])  # (n_alpha, d)
    scores = np.nan_to_num(scores, nan=-1.0)
    best = scores.argmax(0)                                                       # (d,)

    # refit on the full training fold and pick each drug's alpha
    mu = np.nanmean(Yraw, axis=0)
    mu = np.where(np.isnan(mu), np.nanmean(Yraw), mu)
    Yfill = np.where(np.isnan(Yraw), mu, Yraw)
    full = _ridge_path(Xtr, Yfill, Xte, ALPHAS)                                   # (n_alpha, n_te, d)
    return full[best, :, np.arange(d)].T
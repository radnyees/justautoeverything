"""Top-5,000 most variable genes, standardized, ridge regression with a per-drug penalty chosen by
grouped inner CV (maximizing per-drug Pearson on observed AUCs). All alphas/drugs are solved at once
via a thin SVD of the training matrix. Missing training AUCs are filled with the drug's training mean."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(2.5, 6.0, 15)


def _ridge_svd_preds(Xtr, Ytr, Xte, alphas):
    """Return predictions of shape (n_alpha, n_te, n_targets) for ridge with intercept."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    XteV = (Xte - xm) @ Vt.T
    out = np.empty((len(alphas), Xte.shape[0], Ytr.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = (XteV * (s / (s ** 2 + a))) @ UtY + ym
    return out


def _pearson_cols(P, Y, mask):
    """Per-column Pearson between P and Y using only entries where mask is True."""
    r = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        m = mask[:, j]
        if m.sum() < 5:
            continue
        p, y = P[m, j], Y[m, j]
        p, y = p - p.mean(), y - y.mean()
        d = np.sqrt((p ** 2).sum() * (y ** 2).sum())
        r[j] = (p * y).sum() / d if d > 0 else 0.0
    return r


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    mask = ~np.isnan(Yraw)
    Y = y_train.fillna(y_train.mean()).values

    groups = ctx.get("groups_train")
    groups = np.arange(len(Y)) if groups is None else np.asarray(groups)

    # Inner grouped CV: out-of-fold predictions for every alpha and every drug
    oof = np.zeros((len(ALPHAS),) + Y.shape)
    for tr, va in GroupKFold(n_splits=5).split(Xtr, Y, groups):
        Ytr_in = np.where(mask[tr], Yraw[tr], np.nanmean(np.where(mask[tr], Yraw[tr], np.nan), axis=0))
        Ytr_in = np.nan_to_num(Ytr_in, nan=np.nanmean(Yraw))
        oof[:, va, :] = _ridge_svd_preds(Xtr[tr], Ytr_in, Xtr[va], ALPHAS)

    scores = np.stack([_pearson_cols(oof[i], Yraw, mask) for i in range(len(ALPHAS))])
    best = scores.argmax(0)  # per-drug alpha index

    preds_all = _ridge_svd_preds(Xtr, Y, Xte, ALPHAS)
    return preds_all[best, :, np.arange(Y.shape[1])].T
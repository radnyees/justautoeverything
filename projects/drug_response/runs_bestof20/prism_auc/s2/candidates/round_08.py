"""Top-5,000 most variable genes, standardized; ridge regression for all drugs with a PER-DRUG
penalty chosen by grouped inner CV (Pearson r on observed AUCs), computed efficiently via SVD."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(2, 6.5, 19)


def _svd_ridge_preds(Xtr, Ytr, Xte, alphas):
    """Return array (n_alpha, n_te, n_drugs) of ridge predictions (with intercept)."""
    xm = Xtr.mean(0)
    ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    keep = s > 1e-8 * s.max()
    U, s, Vt = U[:, keep], s[keep], Vt[keep]
    XV = (Xte - xm) @ Vt.T            # n_te x k
    UtY = U.T @ (Ytr - ym)            # k x d
    out = np.empty((len(alphas), Xte.shape[0], Ytr.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = XV @ ((s / (s ** 2 + a))[:, None] * UtY) + ym
    return out


def _masked_pearson(P, Y, M):
    """Per-column Pearson r between P and Y using only entries where M is True."""
    w = M.astype(float)
    n = w.sum(0).clip(min=1)
    Yz = np.where(M, Y, 0.0)
    pm = (P * w).sum(0) / n
    ymn = (Yz * w).sum(0) / n
    pc = (P - pm) * w
    yc = (Yz - ymn) * w
    num = (pc * yc).sum(0)
    den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0)) + 1e-12
    return num / den


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    M = ~np.isnan(Yraw)
    Y = y_train.fillna(y_train.mean()).values

    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(Xtr))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))

    # inner grouped CV: out-of-fold predictions for every alpha
    oof = np.zeros((len(ALPHAS),) + Y.shape)
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, Y, groups):
        oof[:, va, :] = _svd_ridge_preds(Xtr[tr], Y[tr], Xtr[va], ALPHAS)
    scores = np.stack([_masked_pearson(oof[i], Yraw, M) for i in range(len(ALPHAS))])  # a x d
    scores = np.nan_to_num(scores, nan=-1.0)
    best = scores.argmax(0)

    full = _svd_ridge_preds(Xtr, Y, Xte, ALPHAS)  # a x n_te x d
    pred = full[best, :, np.arange(Y.shape[1])].T  # n_te x d
    return pred
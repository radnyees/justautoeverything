"""Top-5,000 most variable genes, standardized, multi-output ridge regression with a PER-DRUG
ridge penalty chosen by grouped inner CV (Pearson r on observed AUCs), computed efficiently via SVD.
Missing training AUCs are filled with the drug's training mean for fitting."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _pearson_cols(P, Yraw, mask):
    """Per-column Pearson r between predictions P and observed Yraw, using only observed entries."""
    D = P.shape[1]
    r = np.zeros(D)
    for j in range(D):
        m = mask[:, j]
        if m.sum() < 5:
            continue
        p, y = P[m, j], Yraw[m, j]
        ps, ys = p.std(), y.std()
        if ps < 1e-12 or ys < 1e-12:
            continue
        r[j] = np.mean((p - p.mean()) * (y - y.mean())) / (ps * ys)
    return r


def _ridge_svd_predict(Xtr, Ytr, Xte, alphas):
    """Return predictions for each alpha: array (n_alpha, n_te, D)."""
    xm = Xtr.mean(0)
    ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    XV = (Xte - xm) @ Vt.T
    out = np.empty((len(alphas), Xte.shape[0], Ytr.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = XV @ ((s / (s ** 2 + a))[:, None] * UtY) + ym
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    mask = ~np.isnan(Yraw)
    Y = y_train.fillna(y_train.mean()).values
    Y = np.nan_to_num(Y, nan=0.0)
    n, D = Y.shape

    alphas = np.logspace(2.0, 5.5, 15)

    groups = ctx.get("groups_train", None)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros((len(alphas), n, D))
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, Y, groups):
        oof[:, va, :] = _ridge_svd_predict(Xtr[tr], Y[tr], Xtr[va], alphas)

    scores = np.stack([_pearson_cols(oof[i], Yraw, mask) for i in range(len(alphas))])  # (A, D)
    best = scores.argmax(0)

    preds_all = _ridge_svd_predict(Xtr, Y, Xte, alphas)  # (A, n_te, D)
    return preds_all[best, :, np.arange(D)].T
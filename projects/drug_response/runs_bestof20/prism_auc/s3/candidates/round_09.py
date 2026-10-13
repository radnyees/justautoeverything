"""Top-5,000 most variable genes, standardized, multi-output ridge regression for all drugs
(missing training AUCs filled with the drug's training mean), with a PER-DRUG ridge alpha chosen
by grouped inner CV (per-drug Pearson on observed entries), computed in closed form via SVD."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.array([300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0])


def _ridge_svd_predict(Xa, Ya, Xb, alphas):
    """Return array (n_alpha, n_b, n_drugs) of ridge predictions (with intercept) for each alpha."""
    mu_x = Xa.mean(0)
    mu_y = Ya.mean(0)
    Xc = Xa - mu_x
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ (Ya - mu_y)               # (k, d)
    XbV = (Xb - mu_x) @ Vt.T              # (n_b, k)
    out = []
    for a in alphas:
        d = s / (s ** 2 + a)
        out.append(XbV @ (d[:, None] * UtY) + mu_y)
    return np.stack(out)


def _pearson_cols(P, Y, M):
    """Per-column Pearson r between P and Y using only entries where mask M is True."""
    r = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        m = M[:, j]
        if m.sum() < 5:
            continue
        p, y = P[m, j], Y[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            continue
        r[j] = np.corrcoef(p, y)[0, 1]
    return r


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    M = ~np.isnan(Yraw)
    Y = y_train.fillna(y_train.mean()).values

    # inner grouped CV to choose alpha per drug
    groups = np.asarray(ctx.get("groups_train", np.arange(len(Xtr))))
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros((len(ALPHAS),) + Y.shape)
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, Y, groups):
        Ya = Yraw[tr]
        Ya = np.where(np.isnan(Ya), np.nanmean(Ya, axis=0), Ya)
        Ya = np.where(np.isnan(Ya), np.nanmean(Yraw, axis=0), Ya)
        oof[:, va, :] = _ridge_svd_predict(Xtr[tr], Ya, Xtr[va], ALPHAS)
    scores = np.stack([_pearson_cols(oof[k], Yraw, M) for k in range(len(ALPHAS))])
    scores = np.nan_to_num(scores, nan=-1.0)
    default = int(np.argmax(scores.mean(1)))
    best = np.argmax(scores, axis=0)
    best[np.all(scores == -1.0, axis=0)] = default

    P_all = _ridge_svd_predict(Xtr, Y, Xte, ALPHAS)   # (n_alpha, n_test, d)
    pred = P_all[best, :, np.arange(Y.shape[1])].T    # (n_test, d)
    return pred
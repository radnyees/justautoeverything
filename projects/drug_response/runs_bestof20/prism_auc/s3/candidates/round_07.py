"""Top-5,000 most variable genes, standardized, ridge regression with a PER-DRUG penalty chosen by
grouped inner CV (per-drug Pearson on observed AUCs), computed efficiently via an SVD ridge path."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(2, 6, 17)


def _ridge_path(Xa, Ya, Xb, alphas):
    """Predictions on Xb for each alpha: returns array (n_alpha, n_b, n_targets)."""
    mu = Xa.mean(0)
    ym = Ya.mean(0)
    U, s, Vt = np.linalg.svd(Xa - mu, full_matrices=False)
    UY = U.T @ (Ya - ym)
    Z = (Xb - mu) @ Vt.T
    out = np.empty((len(alphas), Xb.shape[0], Ya.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = (Z * (s / (s ** 2 + a))) @ UY + ym
    return out


def _pearson_cols(P, Y, M):
    """Per-column Pearson r between P and Y over observed mask M."""
    r = np.zeros(P.shape[1])
    for j in range(P.shape[1]):
        m = M[:, j]
        if m.sum() < 5:
            continue
        p, y = P[m, j], Y[m, j]
        p = p - p.mean()
        y = y - y.mean()
        d = np.sqrt((p ** 2).sum() * (y ** 2).sum())
        r[j] = (p * y).sum() / d if d > 0 else 0.0
    return r


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Yraw = y_train.values.astype(float)
    M = ~np.isnan(Yraw)
    Y = y_train.fillna(y_train.mean()).values

    # Grouped inner CV to choose alpha per drug
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(Xtr))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    oof = np.zeros((len(ALPHAS),) + Y.shape)
    for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, Y, groups):
        Ytr_in = y_train.iloc[tr]
        Ytr_in = Ytr_in.fillna(Ytr_in.mean()).fillna(y_train.mean()).values
        oof[:, va, :] = _ridge_path(Xtr[tr], Ytr_in, Xtr[va], ALPHAS)
    scores = np.stack([_pearson_cols(oof[i], Yraw, M) for i in range(len(ALPHAS))])  # (n_alpha, n_drugs)
    best = scores.argmax(0)

    full = _ridge_path(Xtr, Y, Xte, ALPHAS)  # (n_alpha, n_test, n_drugs)
    preds = full[best, :, np.arange(Y.shape[1])].T  # (n_test, n_drugs)
    return preds
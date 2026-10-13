"""Top-5,000 most variable genes, standardized, multi-output ridge (alpha=5000) on per-drug standardized
AUCs, followed by reduced-rank projection of predictions onto the top-k drug-response SVD components
(k chosen by grouped inner CV on the training fold; k=all == plain ridge)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
KS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 100, 200]


def _std_targets(Y):
    mu = np.nanmean(Y, axis=0)
    sd = np.nanstd(Y, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = (Y - mu) / sd
    Z[np.isnan(Z)] = 0.0
    return Z, mu, sd


def _fit_ridge_pred(Xtr, Ztr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), Ztr)
    return m.predict(sc.transform(Xte))


def _drug_basis(Z):
    _, _, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    return Vt


def _project(P, Vt, k):
    if k >= Vt.shape[0]:
        return P
    V = Vt[:k]
    m = P.mean(axis=0)
    return m + (P - m) @ V.T @ V


def _pearson_cols(P, Y):
    rs = []
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j])
        if ok.sum() < 3:
            continue
        p, y = P[ok, j], Y[ok, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return np.mean(rs) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:N_GENES]
    Xtr_all = F_train[genes].values
    Xte_all = F_test[genes].values
    Yraw = y_train.values.astype(float)

    # inner grouped CV to choose rank k
    groups = ctx.get("groups_train", None)
    n = Xtr_all.shape[0]
    if groups is not None and len(np.unique(groups)) >= 4:
        splits = GroupKFold(n_splits=4).split(Xtr_all, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=4, shuffle=True, random_state=0).split(Xtr_all)
    scores = np.zeros(len(KS))
    for itr, iva in splits:
        Zi, _, _ = _std_targets(Yraw[itr])
        Vt_i = _drug_basis(Zi)
        P = _fit_ridge_pred(Xtr_all[itr], Zi, Xtr_all[iva])
        for a, k in enumerate(KS):
            scores[a] += _pearson_cols(_project(P, Vt_i, k), Yraw[iva])
    k_best = KS[int(np.argmax(scores))]

    Z, mu, sd = _std_targets(Yraw)
    Vt = _drug_basis(Z)
    P = _fit_ridge_pred(Xtr_all, Z, Xte_all)
    P = _project(P, Vt, k_best)
    return P * sd + mu
"""Per-drug linear kernel ridge on top-5,000 most variable genes (standardized).
Each drug is fit only on its observed lines (no mean imputation); the ridge penalty is chosen per drug
by inner grouped CV over a grid. That choice is half-weighted toward the mean CV curve across all drugs
to reduce selection noise."""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REL_ALPHAS = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def _krr(Ktt, Kvt, y, alphas):
    """Centered linear kernel ridge; returns predictions (n_val, n_alphas)."""
    r = Ktt.mean(1)
    m = r.mean()
    Kc = Ktt - r[:, None] - r[None, :] + m
    Kv = Kvt - Kvt.mean(1)[:, None] - r[None, :] + m
    ym = y.mean()
    s, U = np.linalg.eigh(Kc)
    s = np.clip(s, 0, None)
    Uy = U.T @ (y - ym)
    coefs = U @ (Uy[:, None] / (s[:, None] + alphas[None, :]))
    return Kv @ coefs + ym


def _pearson_cols(P, y):
    Pc = P - P.mean(0)
    yc = y - y.mean()
    den = np.sqrt((Pc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    return np.nan_to_num((Pc * yc[:, None]).sum(0) / den)


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    K = Xtr @ Xtr.T
    Kte = Xte @ Xtr.T
    n = K.shape[0]
    alphas = REL_ALPHAS * (np.trace(K) / n)

    Y = y_train.values.astype(float)
    obs = ~np.isnan(Y)
    n_drugs = Y.shape[1]

    groups = ctx.get("groups_train")
    groups = np.arange(n) if groups is None else np.asarray(groups)
    folds = list(GroupKFold(n_splits=5).split(Xtr, groups=groups))

    # Inner CV curves per drug
    curves = np.zeros((n_drugs, len(alphas)))
    valid = np.zeros(n_drugs, dtype=bool)
    for d in range(n_drugs):
        oof = np.full((n, len(alphas)), np.nan)
        for tr, va in folds:
            itr = tr[obs[tr, d]]
            iva = va[obs[va, d]]
            if len(itr) < 10 or len(iva) == 0:
                continue
            oof[iva] = _krr(K[np.ix_(itr, itr)], K[np.ix_(iva, itr)], Y[itr, d], alphas)
        ok = ~np.isnan(oof[:, 0])
        if ok.sum() >= 10:
            curves[d] = _pearson_cols(oof[ok], Y[ok, d])
            valid[d] = True

    global_curve = curves[valid].mean(0) if valid.any() else np.zeros(len(alphas))
    default_idx = int(np.argmax(global_curve)) if valid.any() else 2

    preds = np.zeros((Xte.shape[0], n_drugs))
    for d in range(n_drugs):
        idx = np.where(obs[:, d])[0]
        if len(idx) < 3:
            preds[:, d] = np.nanmean(Y[:, d]) if len(idx) else 0.0
            continue
        if valid[d]:
            a_idx = int(np.argmax(0.5 * curves[d] + 0.5 * global_curve))
        else:
            a_idx = default_idx
        preds[:, d] = _krr(K[np.ix_(idx, idx)], Kte[:, idx], Y[idx, d],
                           alphas[a_idx:a_idx + 1])[:, 0]
    return preds
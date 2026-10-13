"""Top-5,000 most variable genes, standardized, plus one-hot OncotreeLineage indicators (scale chosen by
grouped inner CV), one multi-output ridge regression for all drugs (missing AUCs filled with drug mean)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
LIN_SCALES = [0.0, 3.0, 10.0, 30.0]


def _lineage(meta, n):
    if isinstance(meta, pd.DataFrame) and "OncotreeLineage" in meta.columns:
        return meta["OncotreeLineage"].astype(str).fillna("NA").values
    return np.array(["NA"] * n)


def _onehot(lin, cats):
    M = np.zeros((len(lin), len(cats)))
    idx = {c: i for i, c in enumerate(cats)}
    for r, l in enumerate(lin):
        if l in idx:
            M[r, idx[l]] = 1.0
    return M


def _fit_pred(Ftr, lin_tr, Ytr, Fte, lin_te, scale):
    genes = Ftr.var().sort_values(ascending=False).index[:N_GENES]
    sc = StandardScaler().fit(Ftr[genes].values)
    Xtr, Xte = sc.transform(Ftr[genes].values), sc.transform(Fte[genes].values)
    if scale > 0:
        cats = sorted(set(lin_tr))
        Ltr, Lte = _onehot(lin_tr, cats), _onehot(lin_te, cats)
        mu = Ltr.mean(0)
        Xtr = np.hstack([Xtr, scale * (Ltr - mu)])
        Xte = np.hstack([Xte, scale * (Lte - mu)])
    Yf = Ytr.fillna(Ytr.mean()).values
    return Ridge(alpha=ALPHA).fit(Xtr, Yf).predict(Xte)


def _mean_pearson(P, Y):
    rs = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 3:
            continue
        a, b = P[m, j], Y[m, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return np.mean(rs) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    lin_tr = _lineage(ctx.get("meta_train"), len(F_train))
    lin_te = _lineage(ctx.get("meta_test"), len(F_test))

    groups = ctx.get("groups_train")
    n = len(F_train)
    if groups is not None and len(np.unique(groups)) >= 4:
        splits = list(GroupKFold(n_splits=4).split(np.zeros(n), groups=np.asarray(groups)))
    else:
        splits = list(KFold(n_splits=4, shuffle=True, random_state=0).split(np.zeros(n)))

    best_s, best_score = 0.0, -np.inf
    if len(set(lin_tr)) > 1:
        for s in LIN_SCALES:
            P = np.full(y_train.shape, np.nan)
            for tr, va in splits:
                P[va] = _fit_pred(F_train.iloc[tr], lin_tr[tr], y_train.iloc[tr],
                                  F_train.iloc[va], lin_tr[va], s)
            score = _mean_pearson(P, y_train.values)
            if score > best_score:
                best_s, best_score = s, score

    return _fit_pred(F_train, lin_tr, y_train, F_test, lin_te, best_s)
"""Top-5,000 most variable genes, standardized, multi-output ridge with a PER-DRUG penalty chosen by
inner CV on the training fold (efficient leave-one-line-out GCV when every line is its own group,
otherwise grouped K-fold). Missing training AUCs are filled with the drug's training mean."""
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(2.5, 6.0, 15)


def _per_drug_alpha_grouped(X, Y, groups, n_splits=5):
    """Grouped K-fold CV MSE per drug for each alpha; return the best alpha per drug."""
    err = np.zeros((len(ALPHAS), Y.shape[1]))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        ym = Y[tr].mean(0)
        for i, a in enumerate(ALPHAS):
            P = Ridge(alpha=a).fit(Xtr, Y[tr] - ym).predict(Xva) + ym
            err[i] += ((P - Y[va]) ** 2).sum(0)
    return ALPHAS[err.argmin(0)]


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    Xtr_raw, Xte_raw = F_train[genes].values, F_test[genes].values
    Y = y_train.fillna(y_train.mean()).values

    groups = ctx.get("groups_train", None)
    unique_groups = groups is None or len(np.unique(np.asarray(groups))) == len(Y)

    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    if unique_groups:
        # each line is its own group -> leave-one-out == grouped CV; closed-form GCV per drug
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Xtr, Y)
        return model.predict(Xte)

    alphas = _per_drug_alpha_grouped(Xtr_raw, Y, np.asarray(groups))
    preds = np.zeros((len(Xte), Y.shape[1]))
    for a in np.unique(alphas):
        cols = np.where(alphas == a)[0]
        preds[:, cols] = Ridge(alpha=a).fit(Xtr, Y[:, cols]).predict(Xte)
    return preds
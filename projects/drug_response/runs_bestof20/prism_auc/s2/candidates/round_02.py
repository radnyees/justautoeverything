"""Top-5,000 most variable genes, standardized; multi-output ridge with a per-drug regularization
strength chosen by efficient leave-one-out GCV on the training fold (RidgeCV, alpha_per_target).
Missing training AUCs are filled with the drug's training mean; targets standardized per drug."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)

    mu = y_train.mean()
    Y = y_train.fillna(mu).values
    ym, ys = Y.mean(axis=0), Y.std(axis=0)
    ys[ys == 0] = 1.0
    Yz = (Y - ym) / ys

    alphas = np.logspace(2.5, 5.5, 13)  # ~300 .. ~300k around the seed's 5000
    model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(Xtr, Yz)
    return model.predict(Xte) * ys + ym
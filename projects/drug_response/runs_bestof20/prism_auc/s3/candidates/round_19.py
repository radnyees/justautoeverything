"""Top-5,000 most variable genes, standardized, multi-output ridge with a per-drug regularization
strength chosen by efficient leave-one-out (GCV) on the training fold only
(missing training AUCs filled with the drug's training mean)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Y = y_train.fillna(y_train.mean()).values
    alphas = np.logspace(2.5, 6.0, 15)
    model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(Xtr, Y)
    return model.predict(Xte)
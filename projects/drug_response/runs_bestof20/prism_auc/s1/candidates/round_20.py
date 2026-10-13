"""Top-5,000 most variable genes, standardized, multi-output ridge regression with a PER-DRUG
regularization strength chosen by efficient leave-one-out (GCV) inner CV on the training fold
(missing training AUCs filled with the drug's training mean)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    mu = y_train.mean()
    Y = y_train.fillna(mu).values
    alphas = np.logspace(2.5, 5.5, 13)
    model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(Xtr, Y)
    return model.predict(Xte)
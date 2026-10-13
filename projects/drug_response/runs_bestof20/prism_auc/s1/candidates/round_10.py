"""Top-5,000 most variable genes, standardized; ridge regression with a per-drug regularization strength
chosen by efficient leave-one-out CV on the training fold (missing training AUCs filled with drug mean)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    mu = y_train.mean()
    sd = y_train.std().replace(0, 1.0).fillna(1.0)
    Y = ((y_train - mu) / sd).fillna(0.0).values
    alphas = np.logspace(2.5, 6.0, 15)
    model = RidgeCV(alphas=alphas, alpha_per_target=True, fit_intercept=True).fit(Xtr, Y)
    pred = model.predict(Xte)
    return pred * sd.values[None, :] + mu.values[None, :]
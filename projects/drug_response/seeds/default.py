"""Seed: top-5,000 most variable genes, standardized, one multi-output ridge regression for all drugs
(missing training AUCs filled with the drug's training mean)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Y = y_train.fillna(y_train.mean()).values
    model = Ridge(alpha=5000.0).fit(Xtr, Y)
    return model.predict(Xte)

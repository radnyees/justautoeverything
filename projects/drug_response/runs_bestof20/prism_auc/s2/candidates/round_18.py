"""Top-5,000 most variable genes, standardized, multi-output ridge regression for all drugs with a
per-drug ridge penalty chosen by efficient leave-one-out GCV on the training fold
(missing training AUCs filled with the drug's training mean)."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    Y = y_train.fillna(y_train.mean()).values
    # Drugs differ widely in how much expression signal they carry; select shrinkage per drug
    # with closed-form LOO (GCV) inside the training fold only.
    alphas = np.logspace(2.5, 6.5, 17)
    model = RidgeCV(alphas=alphas, alpha_per_target=True).fit(Xtr, Y)
    return model.predict(Xte)
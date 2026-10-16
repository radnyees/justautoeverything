PARAMS = {'n_genes': 5000, 'standardize': False, 'model': 'ridge', 'alpha': 5705.019970965933, 'rf_trees': 500, 'rf_leaf': 3}
"""Seed-derived (independent) Optuna template: the seed (top-variable genes, standardization, one
multi-output ridge) with its constants tunable plus the generic regressor list (ridge, random forest)
(docs/preregistration_phase3.md). PARAMS is prepended by core/search.py."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:PARAMS["n_genes"]]
    Xtr, Xte = F_train[genes].values, F_test[genes].values
    if PARAMS["standardize"]:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Y = y_train.fillna(y_train.mean()).values
    if PARAMS["model"] == "rf":
        m = RandomForestRegressor(n_estimators=PARAMS["rf_trees"], min_samples_leaf=PARAMS["rf_leaf"],
                                  max_features="sqrt", random_state=0, n_jobs=-1)
    else:
        m = Ridge(alpha=PARAMS["alpha"])
    return m.fit(Xtr, Y).predict(Xte)

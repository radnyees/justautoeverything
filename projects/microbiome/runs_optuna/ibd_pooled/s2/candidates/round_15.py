PARAMS = {'level': 'genus', 'norm': 'sqrt_relab', 'prevalence': 0.05, 'study_std': False, 'model': 'rf', 'n_estimators': 1000, 'max_depth': 3, 'C': 0.4397736784270435}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LEVEL_IDX = {"genus": 5, "family": 4}


def featurize(items, ctx):
    X = items.astype(float)
    if PARAMS["level"] != "otu":
        k = LEVEL_IDX[PARAMS["level"]]
        names = [";".join(c.split(";")[:k + 1]) for c in X.columns]
        X = X.T.groupby(np.array(names)).sum().T
    rel = X.div(X.sum(axis=1).clip(lower=1), axis=0)
    norm = PARAMS["norm"]
    if norm == "relab":
        F = rel
    elif norm == "sqrt_relab":
        F = np.sqrt(rel)
    elif norm == "log_relab":
        F = np.log10(rel + 1e-6)
    else:                                            # clr with pseudocount 0.5
        L = np.log(X + 0.5)
        F = L.sub(L.mean(axis=1), axis=0)
    meta = ctx.get("meta")
    if PARAMS["study_std"] and meta is not None and "study" in meta:
        st = meta["study"].reindex(F.index).values
        F = F.copy()
        for s in np.unique(st):
            m = st == s
            F.loc[m] = (F.loc[m] - F.loc[m].mean()) / (F.loc[m].std() + 1e-9)
    return F


def fit_predict(F_train, y_train, F_test, ctx):
    present = (F_train != F_train.min()).mean() if PARAMS["norm"] != "relab" else (F_train > 0).mean()
    cols = F_train.columns[present >= PARAMS["prevalence"]]
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=float(F_train[cols].values.min())).values
    m = PARAMS["model"]
    if m == "rf":
        clf = RandomForestClassifier(n_estimators=PARAMS["n_estimators"], max_features="sqrt",
                                     random_state=0, n_jobs=-1)
    elif m == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=PARAMS["max_depth"], learning_rate=0.05,
                                             max_iter=300, random_state=0)
    else:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(C=PARAMS["C"], penalty="l1" if m == "lr_l1" else "l2",
                                 solver="liblinear" if m == "lr_l1" else "lbfgs", max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]

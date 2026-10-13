PARAMS = {'min_present': 0.5, 'norm': 'none', 'batch_center': False, 'impute': 'halfmin', 'select_k': 400, 'model': 'hgb', 'C': 0.0012057683079698032}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    if PARAMS["norm"] == "median":
        L = L.sub(L.median(axis=1), axis=0)
    elif PARAMS["norm"] == "quantile":                       # rank-based per-sample normalization
        R = L.rank(axis=1, pct=True)
        ref = np.sort(np.nanmedian(np.sort(L.values, axis=1), axis=0))
        L = R.apply(lambda r: pd.Series(np.interp(r.values, np.linspace(0, 1, len(ref)), ref), index=r.index)
                    .where(r.notna()), axis=1)
    meta = ctx.get("meta")
    key = "batch" if (meta is not None and "batch" in meta) else ("site" if meta is not None and "site" in meta else None)
    if PARAMS["batch_center"] and key:
        b = meta[key].reindex(L.index).values
        L = L.copy()
        for v in np.unique(b):
            m = b == v
            L.loc[m] = L.loc[m] - L.loc[m].median()
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= PARAMS["min_present"]]
    Ftr, Fte = F_train[keep], F_test[keep]
    if PARAMS["impute"] == "halfmin":
        fill = Ftr.min() - 1.0
    else:
        fill = Ftr.median()
    Xtr, Xte = Ftr.fillna(fill).values, Fte.fillna(fill).values
    if PARAMS["select_k"]:
        f, _ = f_classif(Xtr, y_train)
        idx = np.argsort(-np.nan_to_num(f))[:PARAMS["select_k"]]
        Xtr, Xte = Xtr[:, idx], Xte[:, idx]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = PARAMS["model"]
    if m == "rf":
        clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    elif m == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    elif m == "svm_linear":
        clf = SVC(kernel="linear", C=PARAMS["C"], probability=False)
        clf.fit(Xtr, y_train)
        return clf.decision_function(Xte)
    else:
        clf = LogisticRegression(C=PARAMS["C"], penalty="l1" if m == "lr_l1" else "l2",
                                 solver="liblinear" if m == "lr_l1" else "lbfgs", max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]

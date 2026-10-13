"""Genus-level relative abundance, percentile-normalized within study (label-free batch harmonization)
+ rank-averaged ensemble of random forest and inner-CV-tuned L2 logistic regression."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta["study"].reindex(items.index).astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return None


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    if studies is None:
        return G.rank(pct=True)
    out = G.copy()
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, pct=True, method="average").values
    return out


def _inner_splits(y, groups, n_splits=5):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values.astype(float)
    Xte = F_test.reindex(columns=cols, fill_value=0).values.astype(float)
    y = np.asarray(y_train)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = None if groups is None else np.asarray(groups)
    try:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=_inner_splits(y, groups),
                                  penalty="l2", scoring="roc_auc", max_iter=2000,
                                  n_jobs=-1, random_state=0)
        lr.fit(Ztr, y)
    except Exception:
        lr = LogisticRegression(C=0.01, max_iter=2000).fit(Ztr, y)
    p_lr = lr.predict_proba(Zte)[:, 1]

    n = len(p_rf)
    return (rankdata(p_rf) + rankdata(p_lr)) / (2.0 * n)
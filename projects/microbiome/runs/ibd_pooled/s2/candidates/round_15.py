"""Genus-level features, harmonized within study (label-free):
 - RF on within-study percentile-ranked relative abundance
 - L2 logistic regression on within-study z-scored genus CLR, with C tuned by
   grouped inner CV scored as WITHIN-study AUROC (matching the harness metric)
Ensemble: each model's test predictions are percentile-ranked WITHIN study
(matching the within-study AUROC scoring), then averaged."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def _meta_studies(F, ctx, key):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    try:
        if hasattr(F, "index") and set(F.index).issubset(set(meta.index)):
            return meta["study"].reindex(F.index).astype(str).values
    except Exception:
        pass
    if len(meta) == len(F):
        return meta["study"].astype(str).values
    return None


def featurize(items, ctx):
    genus = np.array([c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in items.columns])
    counts = items.T.groupby(genus).sum().T.astype(float)          # samples x genera (raw counts)
    rel = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
    logc = np.log(counts + 0.5)
    clr = logc.sub(logc.mean(axis=1), axis=0)

    studies = _study_labels(items, ctx)
    if studies is None:
        studies = np.array(["all"] * len(items))
    P = rel.copy()
    C = clr.copy()
    for s in pd.unique(studies):
        m = studies == s
        P.loc[m] = rel.loc[m].rank(axis=0, pct=True, method="average").values
        block = clr.loc[m].values
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd < 1e-8] = np.inf                                     # constant in this study -> 0
        C.loc[m] = (block - mu) / sd
    P.columns = ["P|" + c for c in P.columns]
    C.columns = ["C|" + c for c in C.columns]
    return pd.concat([P, C], axis=1)


def _inner_splits(y, groups, n_splits=5):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def _within_study_rank(p, studies):
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    if studies is None:
        return rankdata(p) / len(p)
    for s in pd.unique(studies):
        m = studies == s
        out[m] = rankdata(p[m]) / m.sum()
    return out


def _within_study_auc(y, p, studies):
    if studies is None:
        studies = np.array(["all"] * len(y))
    aucs = []
    for s in pd.unique(studies):
        m = studies == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return np.mean(aucs) if aucs else 0.5


def _tune_lr(Z, y, groups, studies, Cs):
    splits = _inner_splits(y, groups)
    oof = np.zeros((len(Cs), len(y)))
    for tr, va in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        for i, C in enumerate(Cs):
            m = LogisticRegression(C=C, penalty="l2", max_iter=2000)
            m.fit(Z[tr], y[tr])
            oof[i, va] = m.decision_function(Z[va])
    scores = [_within_study_auc(y, oof[i], studies) for i in range(len(Cs))]
    return Cs[int(np.argmax(scores))]


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(F_train.columns)
    pcols = [c for c in cols if c.startswith("P|")]
    ccols = [c for c in cols if c.startswith("C|")]
    Fte = F_test.reindex(columns=cols, fill_value=0)
    y = np.asarray(y_train)

    Xtr = F_train[pcols].values.astype(float)
    Xte = Fte[pcols].values.astype(float)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    Ctr = F_train[ccols].values.astype(float)
    Cte = Fte[ccols].values.astype(float)
    sc = StandardScaler().fit(Ctr)
    Ztr, Zte = sc.transform(Ctr), sc.transform(Cte)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = None if groups is None else np.asarray(groups)
    tr_studies = _meta_studies(F_train, ctx, "meta_train")
    if tr_studies is not None and len(tr_studies) != len(y):
        tr_studies = None
    try:
        best_C = _tune_lr(Ztr, y, groups, tr_studies, np.logspace(-4, 0, 9))
    except Exception:
        best_C = 0.01
    lr = LogisticRegression(C=best_C, penalty="l2", max_iter=2000).fit(Ztr, y)
    p_lr = lr.predict_proba(Zte)[:, 1]

    studies = _meta_studies(F_test, ctx, "meta_test")
    if studies is not None and len(studies) != len(p_rf):
        studies = None
    return 0.5 * (_within_study_rank(p_rf, studies) + _within_study_rank(p_lr, studies))
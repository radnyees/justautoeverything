"""Genus-level features, harmonized within study (label-free):
 - RF on within-study percentile-ranked relative abundance
 - inner-CV-tuned L2 logistic regression on within-study z-scored genus CLR
 - beta-diversity kNN score: Aitchison distance to the k nearest training controls minus
   to the k nearest training cases, neighbours taken from the same study only
Ensemble: each model's test predictions are percentile-ranked WITHIN study
(matching the within-study AUROC scoring), then averaged."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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


def _split_studies(F, ctx, key):
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
    A = clr.copy()                                                 # raw CLR for Aitchison distance
    P.columns = ["P|" + c for c in P.columns]
    C.columns = ["C|" + c for c in C.columns]
    A.columns = ["A|" + c for c in A.columns]
    return pd.concat([P, C, A], axis=1)


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


def _knn_contrast(Atr, y, Ate, tr_studies, te_studies, k=10):
    """Mean distance to k nearest training controls minus to k nearest training cases,
    using only training samples from the test sample's own study (fallback: all)."""
    out = np.zeros(len(Ate))
    te_keys = te_studies if te_studies is not None else np.array(["all"] * len(Ate))
    for s in pd.unique(te_keys):
        mte = te_keys == s
        if tr_studies is not None and s != "all":
            mtr = tr_studies == s
            if mtr.sum() < 4 or len(np.unique(y[mtr])) < 2:
                mtr = np.ones(len(y), dtype=bool)
        else:
            mtr = np.ones(len(y), dtype=bool)
        D = cdist(Ate[mte], Atr[mtr])
        ys = y[mtr]
        score = np.zeros(mte.sum())
        for cls, sign in ((0, 1.0), (1, -1.0)):
            Dc = D[:, ys == cls]
            if Dc.shape[1] == 0:
                continue
            kk = min(k, Dc.shape[1])
            near = np.partition(Dc, kk - 1, axis=1)[:, :kk]
            score += sign * near.mean(axis=1)
        out[mte] = score
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(F_train.columns)
    pcols = [c for c in cols if c.startswith("P|")]
    ccols = [c for c in cols if c.startswith("C|")]
    acols = [c for c in cols if c.startswith("A|")]
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
    try:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=_inner_splits(y, groups),
                                  penalty="l2", scoring="roc_auc", max_iter=2000,
                                  n_jobs=-1, random_state=0)
        lr.fit(Ztr, y)
    except Exception:
        lr = LogisticRegression(C=0.01, max_iter=2000).fit(Ztr, y)
    p_lr = lr.predict_proba(Zte)[:, 1]

    studies = _split_studies(F_test, ctx, "meta_test")
    if studies is not None and len(studies) != len(p_rf):
        studies = None
    tr_studies = _split_studies(F_train, ctx, "meta_train")
    if tr_studies is not None and len(tr_studies) != len(y):
        tr_studies = None
    if studies is None or tr_studies is None:
        studies_knn, tr_knn = None, None
    else:
        studies_knn, tr_knn = studies, tr_studies

    Atr = F_train[acols].values.astype(float)
    Ate = Fte[acols].values.astype(float)
    p_knn = _knn_contrast(Atr, y, Ate, tr_knn, studies_knn, k=10)

    return (_within_study_rank(p_rf, studies) + _within_study_rank(p_lr, studies)
            + _within_study_rank(p_knn, studies)) / 3.0
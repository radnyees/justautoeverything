"""Genus-level features, harmonized within study (label-free):
 - RF on within-study percentile-ranked relative abundance
 - inner-CV-tuned L2 logistic regression on within-study z-scored genus CLR
Ensemble: each model's test predictions are percentile-ranked WITHIN study
(matching the within-study AUROC scoring), then blended with a weight chosen by
grouped inner CV on the training fold (within-study AUROC), shrunk toward 0.5."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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
    return np.mean(aucs) if aucs else np.nan


def _fit_rf(X, y, n_estimators=500):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    return rf.fit(X, y)


def _choose_weight(Xp, Zc, y, C, splits, studies):
    """Grouped inner-CV OOF predictions for both models -> best within-study blend weight."""
    oof_rf = np.full(len(y), np.nan)
    oof_lr = np.full(len(y), np.nan)
    for tr, va in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        oof_rf[va] = _fit_rf(Xp[tr], y[tr], 300).predict_proba(Xp[va])[:, 1]
        sc = StandardScaler().fit(Zc[tr])
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Zc[tr]), y[tr])
        oof_lr[va] = lr.predict_proba(sc.transform(Zc[va]))[:, 1]
    ok = ~(np.isnan(oof_rf) | np.isnan(oof_lr))
    if ok.sum() < 20:
        return 0.5
    st = None if studies is None else studies[ok]
    r_rf = _within_study_rank(oof_rf[ok], st)
    r_lr = _within_study_rank(oof_lr[ok], st)
    best_w, best = 0.5, -np.inf
    for w in np.round(np.linspace(0, 1, 11), 2):
        a = _within_study_auc(y[ok], w * r_rf + (1 - w) * r_lr, st)
        if np.isfinite(a) and (a > best + 1e-9 or (abs(a - best) <= 1e-9 and abs(w - 0.5) < abs(best_w - 0.5))):
            best, best_w = a, w
    return 0.5 * best_w + 0.25                                     # shrink halfway toward 0.5


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(F_train.columns)
    pcols = [c for c in cols if c.startswith("P|")]
    ccols = [c for c in cols if c.startswith("C|")]
    Fte = F_test.reindex(columns=cols, fill_value=0)
    y = np.asarray(y_train)

    Xtr = F_train[pcols].values.astype(float)
    Xte = Fte[pcols].values.astype(float)
    rf = _fit_rf(Xtr, y, 500)
    p_rf = rf.predict_proba(Xte)[:, 1]

    Ctr = F_train[ccols].values.astype(float)
    Cte = Fte[ccols].values.astype(float)
    sc = StandardScaler().fit(Ctr)
    Ztr, Zte = sc.transform(Ctr), sc.transform(Cte)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = None if groups is None else np.asarray(groups)
    splits = _inner_splits(y, groups)
    try:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=splits,
                                  penalty="l2", scoring="roc_auc", max_iter=2000,
                                  n_jobs=-1, random_state=0)
        lr.fit(Ztr, y)
        best_C = float(lr.C_[0])
    except Exception:
        best_C = 0.01
        lr = LogisticRegression(C=best_C, max_iter=2000).fit(Ztr, y)
    p_lr = lr.predict_proba(Zte)[:, 1]

    tr_studies = _split_studies(F_train, ctx, "meta_train")
    if tr_studies is not None and len(tr_studies) != len(y):
        tr_studies = None
    try:
        w = _choose_weight(Xtr, Ctr, y, best_C, splits, tr_studies)
    except Exception:
        w = 0.5

    studies = _split_studies(F_test, ctx, "meta_test")
    if studies is not None and len(studies) != len(p_rf):
        studies = None
    return w * _within_study_rank(p_rf, studies) + (1 - w) * _within_study_rank(p_lr, studies)
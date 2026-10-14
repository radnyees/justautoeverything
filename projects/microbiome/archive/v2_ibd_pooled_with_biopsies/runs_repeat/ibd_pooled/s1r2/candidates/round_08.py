"""Genus-level features in two views: (R) within-study percentile ranks of relative abundance for a
random forest, and (C) genus CLR (count pseudocount 0.5) z-scored within study for an L2 logistic
regression (inner grouped CV for C). Both models are trained with per-(study, class) balanced
sample weights so study prevalence is not learnable. Predictions are rank-averaged."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def _study_from_meta(meta, index, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    try:
        s = meta["study"].reindex(index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == n:
            return meta["study"].astype(str).values
        return np.zeros(n, dtype=object)


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    return _study_from_meta(meta, items.index, len(items))


def featurize(items, ctx):
    genus = np.array([c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in items.columns])
    Gc = items.T.groupby(genus).sum().T.astype(float)        # samples x genera (counts)
    rel = Gc.div(Gc.sum(axis=1).clip(lower=1), axis=0)
    logc = np.log(Gc + 0.5)
    clr = logc.sub(logc.mean(axis=1), axis=0)                # centered log-ratio per sample
    studies = _study_labels(items, ctx)
    R = rel.copy()
    C = clr.copy()
    for s in pd.unique(studies):
        m = studies == s
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        R.loc[m] = (rel.loc[m].rank(axis=0, method="average") / max(m.sum(), 1)).values
        sub = clr.loc[m]
        # within-study z-score of CLR: keeps log fold-change scale, removes study offsets
        C.loc[m] = ((sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=0).clip(lower=1e-6)).values
    R.columns = ["R|" + c for c in R.columns]
    C.columns = ["C|" + c for c in C.columns]
    return pd.concat([R, C], axis=1)


def _balanced_weights(studies, y):
    """Weight 1/(n in study x class cell), rescaled to mean 1: each study contributes
    equal case and control mass, removing study-prevalence as a learnable shortcut."""
    y = np.asarray(y)
    df = pd.DataFrame({"s": studies, "y": y})
    cell = df.groupby(["s", "y"])["y"].transform("size").values.astype(float)
    n_cls = df.groupby("s")["y"].transform("nunique").values.astype(float)
    study_n = df.groupby("s")["y"].transform("size").values.astype(float)
    # each study keeps total mass proportional to its size, split equally across its classes
    w = study_n / (n_cls * cell)
    return w * (len(w) / w.sum())


def _within_study_auc(y, p, studies):
    aucs = []
    for s in pd.unique(studies):
        m = studies == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return np.mean(aucs) if aucs else 0.5


def _fit_lr(Xtr, y, w, studies, groups, Xte):
    """L2 logistic regression; C chosen by grouped inner CV on within-study AUROC."""
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1]
    if groups is not None and len(np.unique(groups)) >= 4:
        splits = list(GroupKFold(n_splits=4).split(Xtr, y, groups))
    else:
        splits = list(StratifiedKFold(4, shuffle=True, random_state=0).split(Xtr, y))
    scores = np.zeros(len(Cs))
    for tr, va in splits:
        sc = StandardScaler().fit(Xtr[tr])
        A, B = sc.transform(Xtr[tr]), sc.transform(Xtr[va])
        wi = _balanced_weights(studies[tr], y[tr])
        for k, C in enumerate(Cs):
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(A, y[tr], sample_weight=wi)
            scores[k] += _within_study_auc(y[va], lr.decision_function(B), studies[va])
    C = Cs[int(np.argmax(scores))]
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=2000)
    lr.fit(sc.transform(Xtr), y, sample_weight=w)
    return lr.decision_function(sc.transform(Xte))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    F_te = F_test.reindex(columns=cols, fill_value=0)
    rcols = [c for c in cols if c.startswith("R|")]
    ccols = [c for c in cols if c.startswith("C|")]
    y = np.asarray(y_train).astype(int)
    meta_tr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    studies_tr = _study_from_meta(meta_tr, F_train.index, len(F_train))
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else None
    w = _balanced_weights(studies_tr, y)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(F_train[rcols].values, y, sample_weight=w)
    p_rf = rf.predict_proba(F_te[rcols].values)[:, 1]

    p_lr = _fit_lr(F_train[ccols].values, y, w, studies_tr, groups, F_te[ccols].values)

    n = len(F_te)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n
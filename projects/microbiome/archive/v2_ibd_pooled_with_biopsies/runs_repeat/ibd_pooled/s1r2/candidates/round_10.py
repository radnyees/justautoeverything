"""Genus-level relative abundance, per-study percentile-rank harmonization with zeros mapped to 0
(so 'not detected' is identical across studies); ensemble of a random forest and an L2 logistic
regression (inner grouped CV for C), both trained with per-(study, class) balanced sample weights
so study prevalence is not learnable. Predictions are rank-averaged."""
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
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    vals = np.zeros(G.shape, dtype=float)
    Gv = np.array(G.values, dtype=float, copy=True)
    for s in pd.unique(studies):
        m = np.asarray(studies == s)
        sub = Gv[m]
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        r = np.array(pd.DataFrame(sub).rank(axis=0, method="average").values,
                     dtype=float, copy=True) / max(m.sum(), 1)
        r[sub <= 0] = 0.0                                   # undetected -> 0 in every study
        vals[m] = r
    return pd.DataFrame(vals, index=G.index, columns=G.columns)


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
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = np.array(F_train[cols].values, dtype=float, copy=True)
    Xte = np.array(F_test.reindex(columns=cols, fill_value=0).values, dtype=float, copy=True)
    y = np.asarray(y_train).astype(int)
    meta_tr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    studies_tr = _study_from_meta(meta_tr, F_train.index, len(F_train))
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None and len(groups) == len(y) else None
    w = _balanced_weights(studies_tr, y)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _fit_lr(Xtr, y, w, studies_tr, groups, Xte)

    n = len(Xte)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n
"""Genus-level features, harmonized within study (label-free):
 - RF on within-study percentile-ranked genus relative abundance
 - inner-CV-tuned L2 logistic regression on within-study z-scored CLR at
   genus + lineage-keyed family + order ranks (each rank its own CLR)
Rank-averaged ensemble of the two."""
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


def _clr(counts):
    logc = np.log(counts + 0.5)
    return logc.sub(logc.mean(axis=1), axis=0)


def _zscore_within(df, studies):
    out = df.copy()
    for s in pd.unique(studies):
        m = studies == s
        block = df.loc[m].values
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd < 1e-8] = np.inf                                     # constant in this study -> 0
        out.loc[m] = (block - mu) / sd
    return out


def _lineage_key(col, depth):
    parts = col.split(";")
    parts = parts[:depth] + [""] * max(0, depth - len(parts))
    return ";".join(parts)


def featurize(items, ctx):
    genus = np.array([c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in items.columns])
    counts = items.T.groupby(genus).sum().T.astype(float)          # samples x genera (raw counts)
    rel = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)

    studies = _study_labels(items, ctx)
    if studies is None:
        studies = np.array(["all"] * len(items))

    P = rel.copy()
    for s in pd.unique(studies):
        m = studies == s
        P.loc[m] = rel.loc[m].rank(axis=0, pct=True, method="average").values
    P.columns = ["P|" + c for c in P.columns]

    blocks = [P]
    C = _zscore_within(_clr(counts), studies)
    C.columns = ["C|g|" + c for c in C.columns]
    blocks.append(C)
    for tag, depth in (("f", 5), ("o", 4)):                       # family, order (lineage-keyed)
        keys = np.array([_lineage_key(c, depth) for c in items.columns])
        cnt = items.T.groupby(keys).sum().T.astype(float)
        Z = _zscore_within(_clr(cnt), studies)
        Z.columns = ["C|%s|" % tag + c for c in Z.columns]
        blocks.append(Z)
    return pd.concat(blocks, axis=1)


def _inner_splits(y, groups, n_splits=5):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


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
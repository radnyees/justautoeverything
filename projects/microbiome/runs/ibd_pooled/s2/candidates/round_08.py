"""Genus-level features, harmonized within study (label-free):
 - RF on within-study percentile-ranked relative abundance
 - inner-CV-tuned L2 logistic regression on within-study z-scored genus CLR
 - plus a Gevers 2014 microbial-dysbiosis-index log-ratio feature for both models
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


# Gevers 2014 microbial dysbiosis index taxa (RDP lineage substrings, label-free prior).
_MD_UP = ["f__Enterobacteriaceae", "f__Pasteurellaceae", "f__Veillonellaceae",
          "f__Fusobacteriaceae", "f__Gemellaceae"]
_MD_DOWN = ["o__Bacteroidales", "f__Ruminococcaceae", "f__Lachnospiraceae",
            "f__Erysipelotrichaceae", "f__Bifidobacteriaceae"]


def featurize(items, ctx):
    cols = list(items.columns)
    genus = np.array([c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in cols])
    counts = items.T.groupby(genus).sum().T.astype(float)          # samples x genera (raw counts)
    rel = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
    logc = np.log(counts + 0.5)
    clr = logc.sub(logc.mean(axis=1), axis=0)

    # Microbial dysbiosis index on raw OTU counts: log of summed abundances up vs down.
    lineage = np.array(cols)
    up_mask = np.array([any(t in c for t in _MD_UP) for c in cols])
    down_mask = np.array([any(t in c for t in _MD_DOWN) for c in cols])
    raw = items.astype(float)
    up_sum = raw.loc[:, up_mask].sum(axis=1) if up_mask.any() else pd.Series(0.0, index=items.index)
    down_sum = raw.loc[:, down_mask].sum(axis=1) if down_mask.any() else pd.Series(0.0, index=items.index)
    md_index = np.log((up_sum + 0.5) / (down_sum + 0.5))           # higher = more dysbiotic

    studies = _study_labels(items, ctx)
    if studies is None:
        studies = np.array(["all"] * len(items))
    P = rel.copy()
    C = clr.copy()
    md_z = md_index.copy()
    for s in pd.unique(studies):
        m = studies == s
        P.loc[m] = rel.loc[m].rank(axis=0, pct=True, method="average").values
        block = clr.loc[m].values
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd < 1e-8] = np.inf                                     # constant in this study -> 0
        C.loc[m] = (block - mu) / sd
        mb = md_index.loc[m].values
        msd = mb.std()
        md_z.loc[m] = (mb - mb.mean()) / (msd if msd > 1e-8 else np.inf)
    P.columns = ["P|" + c for c in P.columns]
    C.columns = ["C|" + c for c in C.columns]
    md_df = pd.DataFrame({"M|md_index": md_z.values}, index=items.index)
    return pd.concat([P, C, md_df], axis=1)


def _inner_splits(y, groups, n_splits=5):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(F_train.columns)
    pcols = [c for c in cols if c.startswith("P|")]
    ccols = [c for c in cols if c.startswith("C|")]
    mcols = [c for c in cols if c.startswith("M|")]
    Fte = F_test.reindex(columns=cols, fill_value=0)
    y = np.asarray(y_train)

    # Random forest: percentile features + the dysbiosis index.
    Xtr = F_train[pcols + mcols].values.astype(float)
    Xte = Fte[pcols + mcols].values.astype(float)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # Logistic: genus CLR features (scaled) plus the dysbiosis index given stronger weight
    # via a smaller L2 penalty (feature column rescaled by sqrt(9) ~ 9x weaker shrinkage).
    Ctr = F_train[ccols].values.astype(float)
    Cte = Fte[ccols].values.astype(float)
    sc = StandardScaler().fit(Ctr)
    Ztr, Zte = sc.transform(Ctr), sc.transform(Cte)
    mtr = F_train[mcols].values.astype(float) * 3.0
    mte = Fte[mcols].values.astype(float) * 3.0
    Ztr = np.hstack([Ztr, mtr])
    Zte = np.hstack([Zte, mte])
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
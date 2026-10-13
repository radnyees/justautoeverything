"""log2 + per-sample median centring + per-batch per-protein median centring
+ regress out erythrocyte/platelet contamination indices + drop immunoglobulin groups
+ half-min imputation + L2 LR with C chosen by grouped inner CV."""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Plasma sample-quality marker panels (erythrocyte lysis, platelet activation)
_ERY = {"HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "CAT",
        "PRDX1", "PRDX2", "BLVRB", "SLC4A1", "SPTA1", "SPTB", "ANK1", "EPB42", "GYPA",
        "ALAD", "BPGM", "SELENBP1", "UBA1"}
_PLT = {"PF4", "PF4V1", "PPBP", "ITGA2B", "ITGB3", "TLN1", "VCL", "FLNA", "MYH9", "GP1BA",
        "GP9", "GP5", "TREML1", "THBS1", "F13A1", "TPM4", "ACTN1", "MMRN1", "CFL1", "PFN1",
        "ZYX", "LIMS1", "YWHAZ", "TAGLN2", "ACTB"}

# Immunoglobulin chains (variable/constant regions, J chain, lambda-like); excludes IGF*, IGFBP*,
# IGSF*, IGHMBP2 etc.
_IG_RE = re.compile(r"^(IG[HKL](V.*|C\d*|G\d|A\d|M|E|D|J\d*)|IGLL\d*|JCHAIN|IGJ)$")

_C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            if meta.index.equals(items.index):
                b = meta["batch"]
            elif set(items.index).issubset(set(meta.index)):
                b = meta.loc[items.index, "batch"]
            elif len(meta) == len(items):
                b = pd.Series(meta["batch"].values, index=items.index)
            else:
                return None
            return pd.Series(np.asarray(b), index=items.index).astype(str)
    except Exception:
        return None
    return None


def _genes(col):
    g = str(col).split("|")[0]
    return {t.strip().upper() for t in g.split(";") if t.strip()}


def _is_ig(col):
    g = _genes(col)
    return bool(g) and all(_IG_RE.match(t) for t in g)


def _contamination_index(L, cols):
    """Mean z-score over well-quantified marker proteins (per sample)."""
    sub = L[cols]
    sub = sub.loc[:, sub.notna().mean() >= 0.8]
    if sub.shape[1] < 3:
        return None
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    idx = z.mean(axis=1, skipna=True)
    if idx.isna().any():
        idx = idx.fillna(idx.mean())
    return idx.values


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None and b.nunique() > 1:
        overall = L.median(axis=0)
        out = L.copy()
        for lev in b.unique():
            m = (b == lev).values
            med = L.loc[m].median(axis=0)                  # per-batch protein median (observed only)
            shift = (med - overall).fillna(0.0)
            out.loc[m] = L.loc[m].sub(shift, axis=1)
        L = out

    # label-free removal of pre-analytical contamination signatures
    ery_cols = [c for c in L.columns if _genes(c) & _ERY]
    plt_cols = [c for c in L.columns if _genes(c) & _PLT]
    covs = []
    for cols in (ery_cols, plt_cols):
        ci = _contamination_index(L, cols)
        if ci is not None:
            covs.append(ci - ci.mean())
    if covs:
        C = np.column_stack(covs)
        V = L.values.copy()
        obs = ~np.isnan(V)
        for j in range(V.shape[1]):
            o = obs[:, j]
            if o.sum() < 15:
                continue
            Xo = np.column_stack([np.ones(o.sum()), C[o]])
            beta, *_ = np.linalg.lstsq(Xo, V[o, j], rcond=None)
            V[o, j] = V[o, j] - C[o] @ beta[1:]
        L = pd.DataFrame(V, index=L.index, columns=L.columns)
        L = L.drop(columns=sorted(set(ery_cols) | set(plt_cols)))

    # drop immunoglobulin repertoire groups (individual-specific humoral noise)
    ig_cols = [c for c in L.columns if _is_ig(c)]
    if ig_cols:
        L = L.drop(columns=ig_cols)
    return L


def _prep(F_train, F_test):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def _fit_pred(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _choose_C(F_train, y, groups):
    n_splits = 5
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splits = list(GroupKFold(n_splits=n_splits).split(F_train, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=0).split(F_train, y))
    scores = {c: [] for c in _C_GRID}
    for tr, va in splits:
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xva = _prep(F_train.iloc[tr], F_train.iloc[va])
        for c in _C_GRID:
            p = _fit_pred(Xtr, y[tr], Xva, c)
            scores[c].append(roc_auc_score(y[va], p))
    means = {c: np.mean(v) for c, v in scores.items() if v}
    if not means:
        return 0.05
    best = max(means.values())
    # prefer the strongest regularization within a small tolerance of the best inner AUC
    for c in _C_GRID:
        if c in means and means[c] >= best - 0.01:
            return c
    return 0.05


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    if groups is not None:
        groups = np.asarray(groups)
    C = _choose_C(F_train, y, groups)
    Xtr, Xte = _prep(F_train, F_test)
    return _fit_pred(Xtr, y, Xte, C)
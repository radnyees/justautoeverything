"""Lineage-keyed genus/family/order relative abundance + alpha diversity + sample-location one-hot;
rank-blended ensemble of a prevalence-filtered random forest and a CLR + L2-logistic model, with the
blend weight chosen by subject-grouped inner CV (shrunk toward an equal average). Training samples are
weighted so that every subject contributes equal total weight (near-duplicate multi-site samples)."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

LOCATIONS = ["stool", "terminal ileum", "rectum"]
RARE_DEPTH = 1000
PREFIXES = ["k__", "p__", "c__", "o__", "f__", "g__"]
MIN_PREV = 0.05
PSEUDO = 1e-5
BLEND_GRID = np.linspace(0.0, 1.0, 11)


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, np.log(P), 0.0)
    return -(P * L).sum(axis=1)


def _rarefied_richness(C, depth):
    """Expected number of observed OTUs after subsampling `depth` reads without replacement."""
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        x = C[i][C[i] > 0]
        N = x.sum()
        if N <= 0:
            continue
        d = min(depth, N)
        rest = N - x
        logp_absent = np.full(x.shape, -np.inf)
        ok = rest >= d
        logp_absent[ok] = (gammaln(rest[ok] + 1) - gammaln(rest[ok] - d + 1)
                           - gammaln(N + 1) + gammaln(N - d + 1))
        out[i] = np.sum(1.0 - np.exp(logp_absent))
    return out


def _lineage(col, level):
    """Full taxonomy lineage down to `level` (0=kingdom ... 5=genus), keeping parent context."""
    parts = [p.strip() for p in col.split(";")]
    parts = (parts + PREFIXES[len(parts):])[:6] if len(parts) < 6 else parts[:6]
    return ";".join(parts[:level + 1])


def featurize(items, ctx):
    C = items.values.astype(float)
    depth = C.sum(axis=1)
    rel = items.div(np.clip(depth, 1, None), axis=0)

    blocks = []
    for level, tag in [(5, "G"), (4, "F"), (3, "O")]:
        keys = np.array([tag + "|" + _lineage(c, level) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = blocks[0]

    P = rel.values
    alpha = pd.DataFrame(index=items.index)
    alpha["alpha__shannon_otu"] = _shannon(P)
    alpha["alpha__invsimpson_otu"] = 1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None)
    alpha["alpha__rare_richness"] = _rarefied_richness(C, RARE_DEPTH)
    alpha["alpha__shannon_genus"] = _shannon(G.values)
    return pd.concat(blocks + [alpha], axis=1)


def _location_onehot(meta, n):
    """One-hot sample_location (stool / terminal ileum / rectum); zeros if unavailable."""
    out = np.zeros((n, len(LOCATIONS)))
    if meta is None or "sample_location" not in getattr(meta, "columns", []) or len(meta) != n:
        return out
    loc = meta["sample_location"].astype(str).str.strip().str.lower().values
    for j, name in enumerate(LOCATIONS):
        out[:, j] = (loc == name).astype(float)
    return out


def _clr_block(Atr, Ate):
    """Prevalence-filter (on training fold) then CLR-transform with a small pseudocount."""
    keep = (Atr > 0).mean(axis=0) >= MIN_PREV
    if keep.sum() < 2:
        keep = np.ones(Atr.shape[1], dtype=bool)
    out = []
    for A in (Atr[:, keep], Ate[:, keep]):
        L = np.log(A + PSEUDO)
        out.append(L - L.mean(axis=1, keepdims=True))
    return out


def _inner_cv(y, groups, n_splits=5):
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                .split(np.zeros(len(y)), y))


def _subject_weights(groups, n):
    """Weight 1/(#samples of the subject), normalized to mean 1, so each subject counts equally."""
    if groups is None or len(groups) != n:
        return np.ones(n)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv].astype(float)
    return w * (n / w.sum())


def _make_rf():
    return RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)


def _choose_blend_weight(Xtr, Ztr, y, sw, folds, C):
    """Pick RF weight in the rank blend from grouped out-of-fold predictions on the training fold."""
    oof_rf, oof_lr = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            return 0.5
        rf = _make_rf().fit(Xtr[tr], y[tr], sample_weight=sw[tr])
        lr = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000).fit(
            Ztr[tr], y[tr], sample_weight=sw[tr])
        n = len(te)
        oof_rf[te] = rankdata(rf.predict_proba(Xtr[te])[:, 1]) / n
        oof_lr[te] = rankdata(lr.predict_proba(Ztr[te])[:, 1]) / n
    if len(np.unique(y)) < 2:
        return 0.5
    aucs = [roc_auc_score(y, w * oof_rf + (1 - w) * oof_lr) for w in BLEND_GRID]
    w_best = float(BLEND_GRID[int(np.argmax(aucs))])
    return 0.5 * (w_best + 0.5)                             # shrink toward equal weighting


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    cols = F_train.columns                                  # align test features to training features
    Ftr, Fte = F_train[cols], F_test.reindex(columns=cols, fill_value=0)
    Ltr = _location_onehot(ctx.get("meta_train"), Ftr.shape[0])
    Lte = _location_onehot(ctx.get("meta_test"), Fte.shape[0])
    groups = ctx.get("groups_train")
    groups = None if groups is None else np.asarray(groups)
    folds = _inner_cv(y_train, groups)
    sw = _subject_weights(groups, len(y_train))

    # Random forest on prevalence-filtered relative abundances + alpha + location
    taxa = [c for c in cols if not str(c).startswith("alpha__")]
    prev = (Ftr[taxa].values > 0).mean(axis=0)
    rf_cols = [c for c, p in zip(taxa, prev) if p >= MIN_PREV]
    if len(rf_cols) < 2:
        rf_cols = taxa
    rf_cols = rf_cols + [c for c in cols if str(c).startswith("alpha__")]
    Xtr = np.hstack([Ftr[rf_cols].values, Ltr])
    Xte = np.hstack([Fte[rf_cols].values, Lte])
    rf = _make_rf()
    rf.fit(Xtr, y_train, sample_weight=sw)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # CLR + L2 logistic regression on prevalence-filtered genus and family abundances
    lin_tr, lin_te = [], []
    for tag in ("G|", "F|"):
        bc = [c for c in cols if str(c).startswith(tag)]
        if bc:
            a, b = _clr_block(Ftr[bc].values, Fte[bc].values)
            lin_tr.append(a)
            lin_te.append(b)
    ac = [c for c in cols if str(c).startswith("alpha__")]
    lin_tr += [np.log1p(np.clip(Ftr[ac].values, 0, None)), Ltr]
    lin_te += [np.log1p(np.clip(Fte[ac].values, 0, None)), Lte]
    Ztr, Zte = np.hstack(lin_tr), np.hstack(lin_te)
    sc = StandardScaler().fit(Ztr)
    Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
    lr = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], cv=folds,
                              penalty="l2", scoring="roc_auc", solver="lbfgs", max_iter=5000,
                              n_jobs=-1)
    lr.fit(Ztr, y_train, sample_weight=sw)
    p_lr = lr.predict_proba(Zte)[:, 1]

    w = _choose_blend_weight(Xtr, Ztr, y_train, sw, folds, float(lr.C_[0]))
    n = len(p_rf)
    return w * (rankdata(p_rf) / n) + (1 - w) * (rankdata(p_lr) / n)
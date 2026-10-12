"""Top-5,000 most variable genes, standardized, ridge (alpha=5000) for all drugs, solved in dual form,
plus per-drug drug-target-gene features penalised less (weight w on the target-gene kernel),
w chosen by grouped inner CV on the training fold (w=0 == plain shared ridge)."""
import re
import numpy as np
from sklearn.model_selection import GroupKFold

ALPHA = 5000.0
W_GRID = [0.0, 30.0, 300.0, 3000.0]


def _drug_targets(ctx, drug_ids, gene_index):
    d = ctx["drugs"]
    try:
        if "broad_id" in d.columns and set(drug_ids) <= set(d["broad_id"]):
            d = d.drop_duplicates("broad_id").set_index("broad_id").loc[list(drug_ids)].reset_index()
    except Exception:
        pass
    gpos = {g.upper(): i for i, g in enumerate(gene_index)}
    out = []
    for t in d["target"].tolist()[:len(drug_ids)]:
        if not isinstance(t, str):
            out.append([])
            continue
        toks = [s.strip().upper() for s in re.split(r"[,;|/\s]+", t) if s.strip()]
        out.append(sorted({gpos[s] for s in toks if s in gpos}))
    while len(out) < len(drug_ids):
        out.append([])
    return out


def _predict_all(Xa, Ta, Ya, Xb, Tb, tlist, w_grid):
    """Dual ridge with intercept; returns {w: predictions (n_b, n_drugs)}."""
    mx, mt, my = Xa.mean(0), Ta.mean(0), Ya.mean(0)
    Xa, Xb = Xa - mx, Xb - mx
    Ta, Tb = Ta - mt, Tb - mt
    Yc = Ya - my
    K0 = Xa @ Xa.T
    K0b = Xb @ Xa.T
    n = K0.shape[0]
    Kreg = K0 + ALPHA * np.eye(n)
    base = my + K0b @ np.linalg.solve(Kreg, Yc)
    res = {}
    for w in w_grid:
        P = base.copy()
        if w > 0:
            for j, cols in enumerate(tlist):
                if not cols:
                    continue
                A, B = Ta[:, cols], Tb[:, cols]
                K = Kreg + w * (A @ A.T)
                Kb = K0b + w * (B @ A.T)
                P[:, j] = my[j] + Kb @ np.linalg.solve(K, Yc[:, j])
        res[w] = P
    return res


def _mean_pearson(P, Yobs):
    rs = []
    for j in range(Yobs.shape[1]):
        m = ~np.isnan(Yobs[:, j])
        if m.sum() < 3:
            continue
        p, y = P[m, j], Yobs[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    mu = F_train[genes].values.mean(0)
    sd = F_train[genes].values.std(0)
    sd[sd < 1e-8] = 1.0
    Xtr = (F_train[genes].values - mu) / sd
    Xte = (F_test[genes].values - mu) / sd

    drug_ids = list(y_train.columns)
    tl_all = _drug_targets(ctx, drug_ids, F_train.columns)
    union = sorted({c for cols in tl_all for c in cols})
    Tall_tr = F_train.values[:, union].astype(float) if union else np.zeros((len(F_train), 0))
    Tall_te = F_test.values[:, union].astype(float) if union else np.zeros((len(F_test), 0))
    tm, ts = Tall_tr.mean(0), Tall_tr.std(0)
    keep = ts > 1e-6
    remap = {}
    for k, c in enumerate(union):
        if keep[k]:
            remap[c] = len(remap)
    Ttr = ((Tall_tr - tm) / np.where(keep, ts, 1.0))[:, keep]
    Tte = ((Tall_te - tm) / np.where(keep, ts, 1.0))[:, keep]
    tlist = [[remap[c] for c in cols if c in remap] for cols in tl_all]

    Yobs = y_train.values.astype(float)
    Y = y_train.fillna(y_train.mean()).values

    best_w = 0.0
    if any(tlist) and Ttr.shape[1] > 0:
        groups = np.asarray(ctx.get("groups_train", np.arange(len(F_train))))
        n_splits = min(3, len(np.unique(groups)))
        oof = {w: np.zeros_like(Y) for w in W_GRID}
        for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, groups=groups):
            Yi = Yobs[tr]
            colmean = np.nanmean(Yi, axis=0)
            colmean = np.where(np.isnan(colmean), np.nanmean(Yobs), colmean)
            Yi = np.where(np.isnan(Yi), colmean, Yi)
            res = _predict_all(Xtr[tr], Ttr[tr], Yi, Xtr[va], Ttr[va], tlist, W_GRID)
            for w in W_GRID:
                oof[w][va] = res[w]
        scores = {w: _mean_pearson(oof[w], Yobs) for w in W_GRID}
        best_w = max(W_GRID, key=lambda w: (scores[w], -w))

    res = _predict_all(Xtr, Ttr, Y, Xte, Tte, tlist, [best_w])
    return res[best_w]
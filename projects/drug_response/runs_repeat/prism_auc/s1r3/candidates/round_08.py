"""Top-5,000 variable genes, standardized, ridge (alpha=5000) in kernel (dual) form, with a per-drug,
less-penalized block of the drug's annotated target genes (global block weight chosen by grouped inner CV).
Each drug is fit ONLY on the training lines where its AUC was observed (no mean imputation of missing
responses); drugs that share the same observation mask are solved jointly."""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]


def _drug_targets(ctx, cols, gene_set):
    d = ctx["drugs"].copy()
    if "broad_id" in d.columns and set(cols).issubset(set(d["broad_id"])):
        d = d.drop_duplicates("broad_id").set_index("broad_id").reindex(cols)
    else:
        d = d.reset_index(drop=True)
    out = []
    for i in range(len(cols)):
        t = d["target"].iloc[i] if i < len(d) else None
        if not isinstance(t, str):
            out.append([])
            continue
        gs = [g.strip().upper() for g in re.split(r"[,;|/]", t)]
        gs = sorted({g for g in gs if g in gene_set})
        out.append(gs)
    return out


def _predict_grid(F_train, Y, F_test, targets, s_grid):
    """Y: (n_train, D) array with NaN for unobserved responses."""
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    K = Xtr @ Xtr.T
    Kx = Xte @ Xtr.T
    n_te, D = Kx.shape[0], Y.shape[1]
    obs = ~np.isnan(Y)
    col_mean = np.array([Y[obs[:, j], j].mean() if obs[:, j].any() else 0.0 for j in range(D)])

    # base model (no target block): batch drugs sharing an observation mask
    base = np.tile(col_mean, (n_te, 1))
    mask_groups = {}
    for j in range(D):
        if obs[:, j].sum() >= 3:
            mask_groups.setdefault(obs[:, j].tobytes(), []).append(j)
    for key, js in mask_groups.items():
        m = obs[:, js[0]]
        Km = K[np.ix_(m, m)]
        Yc = Y[m][:, js] - col_mean[js]
        base[:, js] = Kx[:, m] @ np.linalg.solve(Km + ALPHA * np.eye(m.sum()), Yc) + col_mean[js]

    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd
    preds = {}
    for s in s_grid:
        P = base.copy()
        if s > 0 and tg:
            for j, t in enumerate(targets):
                t = [g for g in t if g in tset]
                m = obs[:, j]
                if not t or m.sum() < 3:
                    continue
                A = Ttr[t].values[m]
                B = Tte[t].values
                Kd = K[np.ix_(m, m)] + (s / len(t)) * (A @ A.T)
                Kxd = Kx[:, m] + (s / len(t)) * (B @ A.T)
                yc = Y[m, j] - col_mean[j]
                P[:, j] = Kxd @ np.linalg.solve(Kd + ALPHA * np.eye(m.sum()), yc) + col_mean[j]
        preds[s] = P
    return preds


def _mean_pearson(P, Yobs):
    rs = []
    for j in range(Yobs.shape[1]):
        m = ~np.isnan(Yobs[:, j])
        if m.sum() < 5:
            continue
        a, b = P[m, j], Yobs[m, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return np.mean(rs) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(y_train.columns)
    targets = _drug_targets(ctx, cols, set(F_train.columns))
    Yobs = y_train.values.astype(float)

    # inner grouped CV to choose the target-block weight
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    oof = {s: np.zeros_like(Yobs) for s in S_GRID}
    for tr, te in splits:
        p = _predict_grid(F_train.iloc[tr], Yobs[tr], F_train.iloc[te], targets, S_GRID)
        for s in S_GRID:
            oof[s][te] = p[s]
    scores = {s: _mean_pearson(oof[s], Yobs) for s in S_GRID}
    best = max(S_GRID, key=lambda s: (scores[s], -s))

    return _predict_grid(F_train, Yobs, F_test, targets, [best])[best]
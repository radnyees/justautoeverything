"""Top-5,000 variable genes, standardized, ridge (alpha=5000) in kernel (dual) form, with a per-drug,
less-penalized block of the drug's annotated target genes (global block weight chosen by grouped inner CV).
Each drug is fit ONLY on the training lines where its AUC was observed (no mean imputation of missing
responses); drugs that share the same observation mask are solved jointly.
Target transformation: models are fit both on raw AUC and on rank-based inverse-normal AUC (rescaled to the
drug's mean/sd, which damps the leverage of extreme lines); the blend weight between the two is chosen
jointly with the target-block weight by grouped inner CV (weight 0 = raw-only model)."""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]
W_GRID = [0.0, 0.5, 1.0]


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


def _gauss(Y):
    """Per-drug rank-based inverse-normal transform of observed values, rescaled to the drug's mean/sd."""
    Z = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        v = Y[m, j]
        n = len(v)
        if n < 3 or v.std() < 1e-12:
            Z[m, j] = v
            continue
        z = norm.ppf((rankdata(v) - 0.5) / n)
        z = (z - z.mean()) / z.std()
        Z[m, j] = v.mean() + v.std() * z
    return Z


def _predict_grid(F_train, Ys, F_test, targets, s_grid):
    """Ys: list of (n_train, D) arrays with NaN for unobserved responses (same mask).
    Returns dict keyed by (k, s) for the k-th target version and block weight s."""
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    K = Xtr @ Xtr.T
    Kx = Xte @ Xtr.T
    n_te, D = Kx.shape[0], Ys[0].shape[1]
    obs = ~np.isnan(Ys[0])

    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd

    mask_groups = {}
    for j in range(D):
        if obs[:, j].sum() >= 3:
            mask_groups.setdefault(obs[:, j].tobytes(), []).append(j)

    preds = {}
    for k, Y in enumerate(Ys):
        col_mean = np.array([Y[obs[:, j], j].mean() if obs[:, j].any() else 0.0 for j in range(D)])
        # base model (no target block): batch drugs sharing an observation mask
        base = np.tile(col_mean, (n_te, 1))
        for key, js in mask_groups.items():
            m = obs[:, js[0]]
            Km = K[np.ix_(m, m)]
            Yc = Y[m][:, js] - col_mean[js]
            base[:, js] = Kx[:, m] @ np.linalg.solve(Km + ALPHA * np.eye(m.sum()), Yc) + col_mean[js]
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
            preds[(k, s)] = P
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


def _blend(p, w, s):
    return (1.0 - w) * p[(0, s)] + w * p[(1, s)]


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(y_train.columns)
    targets = _drug_targets(ctx, cols, set(F_train.columns))
    Yobs = y_train.values.astype(float)

    # inner grouped CV to choose the target-block weight and the raw/rank blend weight
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    combos = [(w, s) for w in W_GRID for s in S_GRID]
    oof = {c: np.zeros_like(Yobs) for c in combos}
    for tr, te in splits:
        Ytr = Yobs[tr]
        p = _predict_grid(F_train.iloc[tr], [Ytr, _gauss(Ytr)], F_train.iloc[te], targets, S_GRID)
        for (w, s) in combos:
            oof[(w, s)][te] = _blend(p, w, s)
    scores = {c: _mean_pearson(oof[c], Yobs) for c in combos}
    best_w, best_s = max(combos, key=lambda c: (scores[c], -c[0], -c[1]))

    p = _predict_grid(F_train, [Yobs, _gauss(Yobs)], F_test, targets, [best_s])
    return _blend(p, best_w, best_s)
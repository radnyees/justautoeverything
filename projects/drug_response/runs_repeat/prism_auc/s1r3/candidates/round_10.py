"""Top-5,000 variable genes, standardized, ridge (alpha=5000) in kernel (dual) form, with a per-drug,
less-penalized block of the drug's annotated target genes (global block weight chosen by grouped inner CV).
Each drug is fit ONLY on the training lines where its AUC was observed (no mean imputation of missing
responses); drugs that share the same observation mask are solved jointly.
The per-drug training response can be robustified before fitting (none / MAD-winsorized / rank-gauss);
the transform is chosen jointly with the target-block weight by grouped inner CV."""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]
T_GRID = ["raw", "winsor", "rankgauss"]
WINSOR_C = 3.0


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


def _transform_Y(Y, kind):
    """Column-wise transform of observed training responses (NaN kept)."""
    if kind == "raw":
        return Y
    Z = Y.copy()
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 3:
            continue
        v = Y[m, j]
        if kind == "winsor":
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            if mad < 1e-12:
                continue
            Z[m, j] = np.clip(v, med - WINSOR_C * mad, med + WINSOR_C * mad)
        elif kind == "rankgauss":
            r = rankdata(v)
            Z[m, j] = norm.ppf((r - 0.5) / len(v))
    return Z


def _predict_grid(F_train, Y_raw, F_test, targets, combos):
    """Y_raw: (n_train, D) array with NaN for unobserved responses. combos: list of (transform, s)."""
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    K = Xtr @ Xtr.T
    Kx = Xte @ Xtr.T
    n_te, D = Kx.shape[0], Y_raw.shape[1]
    obs = ~np.isnan(Y_raw)

    mask_groups = {}
    for j in range(D):
        if obs[:, j].sum() >= 3:
            mask_groups.setdefault(obs[:, j].tobytes(), []).append(j)
    # cache the inverse-applied factor per mask group
    solvers = {}
    for key, js in mask_groups.items():
        m = obs[:, js[0]]
        solvers[key] = (m, np.linalg.inv(K[np.ix_(m, m)] + ALPHA * np.eye(m.sum())))

    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd

    preds = {}
    for kind in sorted({c[0] for c in combos}, key=T_GRID.index):
        Y = _transform_Y(Y_raw, kind)
        col_mean = np.array([Y[obs[:, j], j].mean() if obs[:, j].any() else 0.0 for j in range(D)])
        base = np.tile(col_mean, (n_te, 1))
        for key, js in mask_groups.items():
            m, Kinv = solvers[key]
            Yc = Y[m][:, js] - col_mean[js]
            base[:, js] = Kx[:, m] @ (Kinv @ Yc) + col_mean[js]
        for s in [c[1] for c in combos if c[0] == kind]:
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
            preds[(kind, s)] = P
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
    combos = [(k, s) for k in T_GRID for s in S_GRID]

    # inner grouped CV to choose the response transform and target-block weight
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    oof = {c: np.zeros_like(Yobs) for c in combos}
    for tr, te in splits:
        p = _predict_grid(F_train.iloc[tr], Yobs[tr], F_train.iloc[te], targets, combos)
        for c in combos:
            oof[c][te] = p[c]
    # scored against raw observed AUC (the evaluation target)
    scores = {c: _mean_pearson(oof[c], Yobs) for c in combos}
    best = max(combos, key=lambda c: (scores[c], -T_GRID.index(c[0]), -c[1]))

    return _predict_grid(F_train, Yobs, F_test, targets, [best])[best]
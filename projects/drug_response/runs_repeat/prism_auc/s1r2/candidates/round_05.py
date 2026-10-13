"""Shared ridge on top-5,000 variance genes (current best), blended with a per-drug ridge on genes
screened by correlation with that drug's AUC inside the training fold. Screen size, penalty and
blend weight are chosen by grouped inner CV (weight 0 recovers the shared ridge)."""
import numpy as np
from sklearn.model_selection import GroupKFold

N_POOL = 10000
N_BASE = 5000
ALPHA_BASE = 5000.0
KS = (50, 200, 1000)
LAMS = (1.0, 5.0, 25.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _std(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _zcols(D):
    s = D.std(0)
    s[s < 1e-12] = 1.0
    return (D - D.mean(0)) / s


def _mean_pearson(P, Y):
    rs = []
    for j in range(Y.shape[1]):
        m = ~np.isnan(Y[:, j])
        if m.sum() < 3:
            continue
        p, y = P[m, j], Y[m, j]
        if p.std() < 1e-12 or y.std() < 1e-12:
            continue
        rs.append(np.corrcoef(p, y)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def _models(Xtr_raw, Ytr, Xte_raw):
    """Returns ymu, baseline deviations (n_te x D), dict of screened deviations per (k, lam)."""
    Ztr, Zte = _std(Xtr_raw, Xte_raw)
    ymu = np.nan_to_num(np.nanmean(Ytr, 0))
    Yc = np.where(np.isnan(Ytr), ymu, Ytr) - ymu
    n, D = Yc.shape
    B = Ztr[:, :N_BASE]
    base = (Zte[:, :N_BASE] @ B.T) @ np.linalg.solve(B @ B.T + ALPHA_BASE * np.eye(n), Yc)
    S = np.abs(Ztr.T @ Yc)
    kmax = max(KS)
    order = np.argsort(-S, axis=0)[:kmax]
    scr = {(k, l): np.zeros((Zte.shape[0], D)) for k in KS for l in LAMS}
    for j in range(D):
        idx = order[:, j]
        Xa, Xb = Ztr[:, idx], Zte[:, idx]
        for k in KS:
            Xk = Xa[:, :k]
            w, V = np.linalg.eigh(Xk @ Xk.T)
            w = np.clip(w, 0, None)
            Vy = V.T @ Yc[:, j]
            Kt = Xb[:, :k] @ Xk.T
            for l in LAMS:
                scr[(k, l)][:, j] = Kt @ (V @ (Vy / (w + l * k)))
    return ymu, base, scr


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:N_POOL]
    Xtr = F_train[genes].values.astype(float)
    Xte = F_test[genes].values.astype(float)
    Y = y_train.values.astype(float)
    groups = ctx.get("groups_train")
    groups = np.arange(len(Xtr)) if groups is None else np.asarray(groups)

    # grouped inner CV to choose (k, lam, weight)
    configs = [(k, l) for k in KS for l in LAMS]
    oof_b = np.zeros_like(Y)
    oof_s = {c: np.zeros_like(Y) for c in configs}
    for tr, va in GroupKFold(n_splits=3).split(Xtr, groups=groups):
        _, b, s = _models(Xtr[tr], Y[tr], Xtr[va])
        oof_b[va] = _zcols(b)
        for c in configs:
            oof_s[c][va] = _zcols(s[c])
    best, best_score = (configs[0], 0.0), -np.inf
    for c in configs:
        for wt in WEIGHTS:
            sc = _mean_pearson((1 - wt) * oof_b + wt * oof_s[c], Y)
            if sc > best_score + 1e-9:
                best_score, best = sc, (c, wt)
    (c, wt) = best

    ymu, b, s = _models(Xtr, Y, Xte)
    ysd = np.nan_to_num(np.nanstd(Y, 0))
    blend = (1 - wt) * _zcols(b) + wt * _zcols(s[c])
    return ymu + 0.3 * ysd * blend
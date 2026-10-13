"""Shared ridge on top-5,000 variance genes, blended with a per-drug ridge on genes screened by
correlation with that drug's AUC inside the training fold. NEW: the screening correlation of each
drug is smoothed with the mean correlation vector of its MOA/target neighbours (drug-similarity
sharing) to denoise gene selection. Screen size, penalty, smoothing and blend weight are chosen by
grouped inner CV (smoothing 0 / weight 0 recover earlier models)."""
import numpy as np
from sklearn.model_selection import GroupKFold

N_POOL = 10000
N_BASE = 5000
ALPHA_BASE = 5000.0
KS = (50, 200, 1000)
LAMS = (1.0, 5.0, 25.0)
GAMMAS = (0.0, 1.0)
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


def _toks(v):
    if not isinstance(v, str):
        return set()
    return {t.strip().lower() for t in v.split(",") if t.strip()}


def _neighbors(drugs, D):
    """Row-normalized drug neighbour matrix from shared MOA tokens or overlapping target sets."""
    A = np.zeros((D, D))
    if drugs is None or len(drugs) != D:
        return A, np.zeros(D, bool)
    moa = [_toks(v) for v in drugs["moa"]] if "moa" in drugs.columns else [set()] * D
    tg = [_toks(v) for v in drugs["target"]] if "target" in drugs.columns else [set()] * D
    for i in range(D):
        for j in range(i + 1, D):
            share = bool(moa[i] & moa[j])
            if not share and tg[i] and tg[j]:
                share = len(tg[i] & tg[j]) / len(tg[i] | tg[j]) >= 0.5
            if share:
                A[i, j] = A[j, i] = 1.0
    deg = A.sum(1)
    return A / np.maximum(deg, 1.0)[:, None], deg > 0


def _models(Xtr_raw, Ytr, Xte_raw, An, has_nb):
    """Returns ymu, baseline deviations (n_te x D), dict of screened deviations per (k, lam, g)."""
    Ztr, Zte = _std(Xtr_raw, Xte_raw)
    ymu = np.nan_to_num(np.nanmean(Ytr, 0))
    Yc = np.where(np.isnan(Ytr), ymu, Ytr) - ymu
    n, D = Yc.shape
    B = Ztr[:, :N_BASE]
    base = (Zte[:, :N_BASE] @ B.T) @ np.linalg.solve(B @ B.T + ALPHA_BASE * np.eye(n), Yc)
    ysd = Yc.std(0)
    ysd[ysd < 1e-12] = 1.0
    C = (Ztr.T @ Yc) / (n * ysd)  # genes x drugs correlations
    Cnb = C @ An.T
    kmax = max(KS)
    orders = {}
    for g in GAMMAS:
        S = np.abs((C + g * Cnb) / (1.0 + g)) if g > 0 else np.abs(C)
        orders[g] = np.argsort(-S, axis=0)[:kmax]
    scr = {(k, l, g): np.zeros((Zte.shape[0], D)) for k in KS for l in LAMS for g in GAMMAS}
    for j in range(D):
        for g in GAMMAS:
            if g > 0 and not has_nb[j]:
                for k in KS:
                    for l in LAMS:
                        scr[(k, l, g)][:, j] = scr[(k, l, 0.0)][:, j]
                continue
            idx = orders[g][:, j]
            Xa, Xb = Ztr[:, idx], Zte[:, idx]
            for k in KS:
                Xk = Xa[:, :k]
                w, V = np.linalg.eigh(Xk @ Xk.T)
                w = np.clip(w, 0, None)
                Vy = V.T @ Yc[:, j]
                Kt = Xb[:, :k] @ Xk.T
                for l in LAMS:
                    scr[(k, l, g)][:, j] = Kt @ (V @ (Vy / (w + l * k)))
    return ymu, base, scr


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:N_POOL]
    Xtr = F_train[genes].values.astype(float)
    Xte = F_test[genes].values.astype(float)
    Y = y_train.values.astype(float)
    D = Y.shape[1]
    groups = ctx.get("groups_train")
    groups = np.arange(len(Xtr)) if groups is None else np.asarray(groups)

    drugs = ctx.get("drugs")
    try:
        if drugs is not None and "broad_id" in drugs.columns:
            d2 = drugs.drop_duplicates("broad_id").set_index("broad_id").reindex(list(y_train.columns))
            if d2["moa"].notna().sum() > 0:
                drugs = d2
            drugs = drugs.reset_index(drop=True)
    except Exception:
        pass
    An, has_nb = _neighbors(drugs, D)

    # grouped inner CV to choose (k, lam, gamma, weight)
    configs = [(k, l, g) for k in KS for l in LAMS for g in GAMMAS]
    oof_b = np.zeros_like(Y)
    oof_s = {c: np.zeros_like(Y) for c in configs}
    for tr, va in GroupKFold(n_splits=3).split(Xtr, groups=groups):
        _, b, s = _models(Xtr[tr], Y[tr], Xtr[va], An, has_nb)
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

    ymu, b, s = _models(Xtr, Y, Xte, An, has_nb)
    ysd = np.nan_to_num(np.nanstd(Y, 0))
    blend = (1 - wt) * _zcols(b) + wt * _zcols(s[c])
    return ymu + 0.3 * ysd * blend
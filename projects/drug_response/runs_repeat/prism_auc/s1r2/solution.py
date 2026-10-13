"""Shared ridge on top-5,000 variance genes, blended with a per-drug ridge on genes screened by
correlation with that drug's AUC inside the training fold. Screening variants: raw correlation,
correlation smoothed with MOA/target neighbours, a residual variant that screens and fits the
per-drug ridge on the shared ridge's training residuals alpha (K + alpha I)^-1 y and adds the
result to the base prediction, and (NEW) a multi-task-target variant "mt" that keeps the
MOA-smoothed screen but fits the ridge to the drug's standardized AUC averaged with its MOA
neighbours' mean standardized AUC (shares the eigendecomposition with "moa", so nearly free).
Besides hard top-k screens, a soft screen weights each of the top-1000 genes by
(|r| / mean|r|)^p. The (screen, penalty, variant, blend weight) option is chosen PER DRUG, shrunk
toward the global optimum, with the shrinkage chosen by cross-fitting over inner folds.
Prediction-level drug sharing: each drug's z-scored prediction is mixed with the mean prediction
of its neighbour drugs, with neighbour type and eta chosen by cross-fitting."""
import warnings

import numpy as np
from sklearn.model_selection import GroupKFold

N_POOL = 10000
N_BASE = 5000
ALPHA_BASE = 5000.0
SCREENS = ((50, 0), (200, 0), (1000, 0), (1000, 1), (1000, 2))  # (k, soft-weight power p)
LAMS = (1.0, 5.0, 25.0)
SCREEN_VARIANTS = ("raw", "moa", "res")
VARIANTS = ("raw", "moa", "res", "mt")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRATS = [("all", 1.0)] + [(m, s) for m in ("w", "all") for s in (0.0, 0.25, 0.5, 0.75)]
SHARES = [(None, 0.0)] + [(k, e) for k in ("moa", "corr") for e in (0.15, 0.3, 0.5)]
CORR_THR = 0.5
CORR_TOP = 10


def _std(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _zcols(D):
    s = D.std(0)
    s[s < 1e-12] = 1.0
    return (D - D.mean(0)) / s


def _colr(P, Y):
    """Per-column Pearson r ignoring NaNs in Y; NaN where undefined."""
    M = ~np.isnan(Y)
    cnt = M.sum(0).astype(float)
    Yf = np.where(M, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = (P * M).sum(0) / cnt
        ym = Yf.sum(0) / cnt
        pc = (P - pm) * M
        yc = (Yf - ym) * M
        den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
        r = (pc * yc).sum(0) / den
    r[(cnt < 3) | ~(den > 1e-12)] = np.nan
    return r


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis)


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


def _corr_neighbors(Y):
    """Row-normalized neighbour matrix from training AUC profile correlations (r >= CORR_THR)."""
    D = Y.shape[1]
    mu = np.nan_to_num(_nanmean(Y, 0))
    Yf = np.where(np.isnan(Y), mu, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(Yf.T)
    R = np.nan_to_num(R)
    np.fill_diagonal(R, 0.0)
    W = np.zeros((D, D))
    for j in range(D):
        idx = np.argsort(-R[j])[:CORR_TOP]
        idx = idx[R[j, idx] >= CORR_THR]
        W[j, idx] = R[j, idx]
    s = W.sum(1)
    return W / np.maximum(s, 1e-12)[:, None], s > 0


def _smooth(Z, W, has, eta):
    if eta <= 0:
        return Z
    Wz = Z @ W.T
    return np.where(has[None, :], (1 - eta) * Z + eta * Wz, Z)


def _models(Xtr_raw, Ytr, Xte_raw, An, has_nb):
    """Returns ymu, baseline deviations (n_te x D), dict of screened deviations per (screen, lam, variant)."""
    Ztr, Zte = _std(Xtr_raw, Xte_raw)
    ymu = np.nan_to_num(np.nanmean(Ytr, 0))
    Yc = np.where(np.isnan(Ytr), ymu, Ytr) - ymu
    n, D = Yc.shape
    B = Ztr[:, :N_BASE]
    Acoef = np.linalg.solve(B @ B.T + ALPHA_BASE * np.eye(n), Yc)
    base = (Zte[:, :N_BASE] @ B.T) @ Acoef
    Res = ALPHA_BASE * Acoef  # training residuals of the shared kernel ridge
    Res = Res - Res.mean(0)

    # multi-task target: drug's standardized AUC averaged with its MOA neighbours' mean
    ysd = Yc.std(0)
    ysd[ysd < 1e-12] = 1.0
    Yz = Yc / ysd
    Ymt = 0.5 * (Yz + Yz @ An.T) * ysd

    def _corr(T):
        sd = T.std(0)
        sd[sd < 1e-12] = 1.0
        return (Ztr.T @ T) / (n * sd)  # genes x drugs correlations

    C = _corr(Yc)
    Cnb = C @ An.T
    S = {"raw": np.abs(C), "moa": np.abs((C + Cnb) / 2.0), "res": np.abs(_corr(Res))}
    targets = {"raw": Yc, "moa": Yc, "res": Res, "mt": Ymt}
    kmax = max(k for k, _ in SCREENS)
    orders = {v: np.argsort(-S[v], axis=0)[:kmax] for v in SCREEN_VARIANTS}
    scr = {(s, l, v): np.zeros((Zte.shape[0], D)) for s in SCREENS for l in LAMS for v in VARIANTS}
    for j in range(D):
        for v in SCREEN_VARIANTS:
            if v == "moa" and not has_nb[j]:
                for s in SCREENS:
                    for l in LAMS:
                        scr[(s, l, "moa")][:, j] = scr[(s, l, "raw")][:, j]
                        scr[(s, l, "mt")][:, j] = scr[(s, l, "raw")][:, j]
                continue
            idx = orders[v][:, j]
            Xa, Xb = Ztr[:, idx], Zte[:, idx]
            sv = S[v][idx, j]
            off = base[:, j] if v == "res" else 0.0
            outs = [v, "mt"] if v == "moa" else [v]
            for (k, p) in SCREENS:
                Xk, Xbk = Xa[:, :k], Xb[:, :k]
                if p > 0:
                    wv = sv[:k] / max(sv[:k].mean(), 1e-12)
                    wv = wv ** p
                    wv = wv / wv.mean()
                    Kk = (Xk * wv) @ Xk.T
                    Kt = (Xbk * wv) @ Xk.T
                else:
                    Kk = Xk @ Xk.T
                    Kt = Xbk @ Xk.T
                w, V = np.linalg.eigh(Kk)
                w = np.clip(w, 0, None)
                KtV = Kt @ V
                for vv in outs:
                    Vy = V.T @ targets[vv][:, j]
                    for l in LAMS:
                        scr[((k, p), l, vv)][:, j] = off + KtV @ (Vy / (w + l * k))
    return ymu, base, scr


def _select(R, strat, opts):
    """Per-drug option choice from an (n_opt x D) Pearson matrix under a shrinkage strategy."""
    mode, s = strat
    rbar = _nanmean(R, 1)
    rbar = np.where(np.isnan(rbar), -np.inf, rbar)
    g = int(np.argmax(rbar))
    if mode == "w":
        allowed = np.array([o for o in range(len(opts)) if opts[o][0] == opts[g][0]])
    else:
        allowed = np.arange(len(opts))
    Ra = R[allowed]
    sc = (1 - s) * Ra + s * rbar[allowed, None]
    sc = np.where(np.isnan(sc), -np.inf, sc)
    ch = allowed[np.argmax(sc, 0)]
    ch[~np.isfinite(sc.max(0))] = g
    return ch


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

    # grouped inner CV out-of-fold predictions for every (screen, lam, variant)
    configs = [(s, l, v) for s in SCREENS for l in LAMS for v in VARIANTS]
    oof_b = np.zeros_like(Y)
    oof_s = {c: np.zeros_like(Y) for c in configs}
    vas = []
    for tr, va in GroupKFold(n_splits=3).split(Xtr, groups=groups):
        vas.append(va)
        _, b, s = _models(Xtr[tr], Y[tr], Xtr[va], An, has_nb)
        oof_b[va] = _zcols(b)
        for c in configs:
            oof_s[c][va] = _zcols(s[c])

    # per-option, per-drug Pearson on full OOF and on cross-fit splits
    opts = [(c, wt) for c in configs for wt in WEIGHTS]
    nO, nF = len(opts), len(vas)
    Rfull = np.zeros((nO, D))
    Rtr = np.zeros((nF, nO, D))
    Rte = np.zeros((nF, nO, D))
    inmask = []
    for va in vas:
        m = np.zeros(len(Y), bool)
        m[va] = True
        inmask.append(m)
    for o, (c, wt) in enumerate(opts):
        P = (1 - wt) * oof_b + wt * oof_s[c]
        Rfull[o] = _colr(P, Y)
        for f, m in enumerate(inmask):
            Rtr[f, o] = _colr(P[~m], Y[~m])
            Rte[f, o] = _colr(P[m], Y[m])

    # choose the per-drug shrinkage strategy by cross-fitting
    best_strat, best_score = STRATS[0], -np.inf
    for strat in STRATS:
        vals = []
        for f in range(nF):
            ch = _select(Rtr[f], strat, opts)
            r = Rte[f][ch, np.arange(D)]
            if np.any(~np.isnan(r)):
                vals.append(np.nanmean(r))
        sc = float(np.mean(vals)) if vals else -np.inf
        if sc > best_score + 1e-9:
            best_score, best_strat = sc, strat
    choice = _select(Rfull, best_strat, opts)

    # choose prediction-level drug sharing (neighbour type, eta) by cross-fitting
    Pz, Wc = [], []
    for f, m in enumerate(inmask):
        ch = _select(Rtr[f], best_strat, opts)
        P = np.zeros((int(m.sum()), D))
        for j in range(D):
            c, wt = opts[ch[j]]
            P[:, j] = (1 - wt) * oof_b[m, j] + wt * oof_s[c][m, j]
        Pz.append(_zcols(P))
        Wc.append(_corr_neighbors(Y[~m]))
    best_share, best_ss = SHARES[0], -np.inf
    for kind, eta in SHARES:
        vals = []
        for f, m in enumerate(inmask):
            if kind is None:
                Ps = Pz[f]
            else:
                W, h = (An, has_nb) if kind == "moa" else Wc[f]
                Ps = _smooth(Pz[f], W, h, eta)
            r = _colr(Ps, Y[m])
            if np.any(~np.isnan(r)):
                vals.append(np.nanmean(r))
        sc = float(np.mean(vals)) if vals else -np.inf
        if sc > best_ss + 1e-9:
            best_ss, best_share = sc, (kind, eta)

    ymu, b, s = _models(Xtr, Y, Xte, An, has_nb)
    ysd = np.nan_to_num(np.nanstd(Y, 0))
    zb = _zcols(b)
    zs = {}
    blend = np.zeros((Xte.shape[0], D))
    for j in range(D):
        c, wt = opts[choice[j]]
        if c not in zs:
            zs[c] = _zcols(s[c])
        blend[:, j] = (1 - wt) * zb[:, j] + wt * zs[c][:, j]

    kind, eta = best_share
    if kind is not None and blend.shape[0] > 1:
        W, h = (An, has_nb) if kind == "moa" else _corr_neighbors(Y)
        blend = _smooth(_zcols(blend), W, h, eta)
    return ymu + 0.3 * ysd * blend
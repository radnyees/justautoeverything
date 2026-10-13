"""Top-5,000 most variable genes, standardized; kernel ridge for all drugs in dual form with
a blended kernel K = (1-v)*linear + v*s*centered-RBF (s matches traces), plus per-drug drug-target-gene
features penalised less (weight w on the target-gene kernel). Each drug is fitted only on the lines where
it was measured (no imputation).
Sparse per-drug biomarker model: for each drug, the top-m genes (out of the 10,000 most variable) by
|correlation| with its training AUCs are selected (training lines only) and a small ridge is fitted on them;
optionally bagged (selection + ridge repeated on row subsamples and averaged) to stabilise the noisy
selection. Its prediction is blended in as (1-lam)*KRR + lam*sparse (lam=0 == pure kernel ridge).
Post-hoc sharing: each drug's prediction is augmented with b x the
mean standardized prediction of drugs sharing an MOA label, plus c x the correlation-weighted mean
standardized prediction of its top-k most response-correlated drugs (correlations from training labels
only), both rescaled to the drug's AUC sd.
Optional robust target transformation: training AUCs are winsorized per drug at median +/- k*1.4826*MAD
(training lines only) before fitting / sharing statistics; inner-CV scoring uses raw AUCs.
(k, alpha, v, w, lam, bag, b, c) chosen jointly by grouped inner CV on the training fold (b=c=0 == no
sharing, k=None == no clipping, bag=0 == single sparse fit)."""
import re
import numpy as np
import pandas as pd
from scipy.linalg import solve as _solve
from sklearn.model_selection import GroupKFold

ALPHA_DEFAULT = 5000.0
A_GRID = [2000.0, 5000.0, 12500.0]
W_GRID = [0.0, 30.0, 300.0, 3000.0]
V_GRID = [0.0, 0.5, 0.9]
B_GRID = [0.0, 0.25, 0.5, 1.0]
C_GRID = [0.0, 0.25, 0.5, 1.0]
K_GRID = [None, 2.5]
L_GRID = [0.0, 0.2, 0.4]
K_NBR = 10
SP_GENES = 10000
SP_M = 50
SP_ALPHA = 1.0
BAG_N = 8
BAG_FRAC = 0.8


def _aligned_drugs(ctx, drug_ids):
    d = ctx["drugs"]
    try:
        if "broad_id" in d.columns and set(drug_ids) <= set(d["broad_id"]):
            d = d.drop_duplicates("broad_id").set_index("broad_id").loc[list(drug_ids)].reset_index()
    except Exception:
        pass
    return d


def _drug_targets(ctx, drug_ids, gene_index):
    d = _aligned_drugs(ctx, drug_ids)
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


def _moa_matrix(ctx, drug_ids):
    """N[k, j] = 1/|nbrs(j)| if drug k shares an MOA label with drug j (k != j), else 0."""
    nd = len(drug_ids)
    N = np.zeros((nd, nd))
    try:
        d = _aligned_drugs(ctx, drug_ids)
        moas = d["moa"].tolist()[:nd] if "moa" in d.columns else []
    except Exception:
        moas = []
    labs = []
    for m in moas:
        if isinstance(m, str):
            labs.append({s.strip().lower() for s in re.split(r"[,;|]+", m) if s.strip()})
        else:
            labs.append(set())
    while len(labs) < nd:
        labs.append(set())
    for j in range(nd):
        nb = [k for k in range(nd) if k != j and labs[j] and (labs[j] & labs[k])]
        if nb:
            N[nb, j] = 1.0 / len(nb)
    return N


def _corr_matrix(Y, k=K_NBR):
    """N2[i, j] = r_ij / sum(r) over the top-k positively correlated drugs i of drug j (from labels)."""
    nd = Y.shape[1]
    C = pd.DataFrame(Y).corr(min_periods=20).values
    C = np.where(np.isnan(C), 0.0, C)
    np.fill_diagonal(C, 0.0)
    C = np.maximum(C, 0.0)
    N2 = np.zeros((nd, nd))
    for j in range(nd):
        idx = np.argsort(-C[:, j])[:k]
        idx = idx[C[idx, j] > 0]
        if len(idx):
            N2[idx, j] = C[idx, j] / C[idx, j].sum()
    return N2


def _winsor(Y, k):
    """Per-drug clip at median +/- k * robust sd (1.4826*MAD, fallback to sd); NaN preserved."""
    if k is None:
        return Y
    with np.errstate(all="ignore"):
        med = np.nanmedian(Y, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(Y - med), axis=0)
        sd = np.nanstd(Y, axis=0)
    s = np.where(np.isfinite(mad) & (mad > 1e-8), mad, sd)
    ok = np.isfinite(med) & np.isfinite(s) & (s > 1e-8)
    lo = np.where(ok, med - k * s, -np.inf)
    hi = np.where(ok, med + k * s, np.inf)
    return np.clip(Y, lo, hi)


def _share(P, my, sdy, N, b, N2=None, c=0.0):
    if (b == 0 or not N.any()) and (c == 0 or N2 is None or not N2.any()):
        return P
    Z = (P - my) / sdy
    out = P
    if b != 0 and N.any():
        out = out + b * (Z @ N) * sdy
    if c != 0 and N2 is not None and N2.any():
        out = out + c * (Z @ N2) * sdy
    return out


def _center_kernel(K, Kb):
    cm = K.mean(0)
    tot = K.mean()
    Kc = K - cm[None, :] - cm[:, None] + tot
    Kbc = Kb - Kb.mean(1)[:, None] - cm[None, :] + tot
    return Kc, Kbc


def _psolve(A, b):
    try:
        return _solve(A, b, assume_a="pos", check_finite=False)
    except Exception:
        return np.linalg.solve(A, b)


def _ystats(Ya):
    obs = ~np.isnan(Ya)
    my = np.nanmean(np.where(obs, Ya, np.nan), axis=0) if obs.any() else np.zeros(Ya.shape[1])
    gm = np.nanmean(Ya) if obs.any() else 0.0
    my = np.where(np.isnan(my), gm, my)
    sdy = np.nanstd(np.where(obs, Ya, np.nan), axis=0) if obs.any() else np.ones(Ya.shape[1])
    sdy = np.where(np.isnan(sdy) | (sdy < 1e-8), 1.0, sdy)
    return obs, my, sdy


def _sparse_predict(Xa, Ya, Xb, m=SP_M, alpha=SP_ALPHA):
    """Per-drug ridge on the top-m genes by |corr| with the drug's observed (training) AUCs."""
    obs, my, _ = _ystats(Ya)
    Yc = np.where(obs, Ya - my, 0.0)
    mx = Xa.mean(0)
    sx = Xa.std(0)
    sx[sx < 1e-8] = 1.0
    Xa_ = (Xa - mx) / sx
    Xb_ = (Xb - mx) / sx
    S = np.abs(Xa_.T @ Yc)
    m = min(m, S.shape[0])
    top = np.argpartition(-S, m - 1, axis=0)[:m]
    P = np.tile(my, (Xb.shape[0], 1))
    for j in range(Ya.shape[1]):
        o = obs[:, j]
        no = int(o.sum())
        if no < 5:
            continue
        idx = top[:, j]
        A = Xa_[np.ix_(o, idx)]
        ac = A.mean(0)
        A = A - ac
        yj = Yc[o, j]
        beta = _psolve(A.T @ A + alpha * no * np.eye(len(idx)), A.T @ yj)
        P[:, j] = my[j] + (Xb_[:, idx] - ac) @ beta
    return P


def _sparse_bagged(Xa, Ya, Xb, n_bag=BAG_N, frac=BAG_FRAC, seed=0):
    """Average of sparse per-drug models refitted (selection + ridge) on row subsamples."""
    rng = np.random.RandomState(seed)
    n = Xa.shape[0]
    k = max(10, int(round(frac * n)))
    if k >= n:
        return _sparse_predict(Xa, Ya, Xb)
    P = np.zeros((Xb.shape[0], Ya.shape[1]))
    for _ in range(n_bag):
        idx = np.sort(rng.choice(n, k, replace=False))
        P += _sparse_predict(Xa[idx], Ya[idx], Xb)
    return P / n_bag


def _predict_all(Xa, Ra, Ta, Ya, Xb, Rb, Tb, tlist, combos):
    """Dual kernel ridge with intercept; Ra/Rb = raw RBF kernels (a x a, b x a).
    Ya may contain NaN: each drug is fitted on its observed lines only.
    combos: list of (alpha, v, w). Returns {(alpha, v, w): predictions (n_b, n_drugs)}."""
    obs, my, _ = _ystats(Ya)
    Yc = np.where(obs, Ya - my, 0.0)
    full = obs.all(0)
    part = np.where(~full)[0]

    mx, mt = Xa.mean(0), Ta.mean(0)
    Xa, Xb = Xa - mx, Xb - mx
    Ta, Tb = Ta - mt, Tb - mt
    Klin = Xa @ Xa.T
    Kblin = Xb @ Xa.T
    Rc, Rbc = _center_kernel(Ra, Rb)
    s = np.trace(Klin) / max(np.trace(Rc), 1e-12)
    n = Klin.shape[0]
    nb = Xb.shape[0]
    res = {}
    for v in sorted({c[1] for c in combos}):
        K0 = (1.0 - v) * Klin + v * s * Rc
        K0b = (1.0 - v) * Kblin + v * s * Rbc
        for a in sorted({c[0] for c in combos if c[1] == v}):
            Kreg = K0 + a * np.eye(n)
            base = np.tile(my, (nb, 1))
            if full.any():
                base[:, full] = my[full] + K0b @ _psolve(Kreg, Yc[:, full])
            for j in part:
                o = obs[:, j]
                if o.sum() < 3:
                    continue
                base[:, j] = my[j] + K0b[:, o] @ _psolve(Kreg[np.ix_(o, o)], Yc[o, j])
            for w in sorted({c[2] for c in combos if c[0] == a and c[1] == v}):
                P = base.copy()
                if w > 0:
                    for j, cols in enumerate(tlist):
                        if not cols:
                            continue
                        o = obs[:, j]
                        if o.sum() < 3:
                            continue
                        A, B = Ta[o][:, cols], Tb[:, cols]
                        K = Kreg[np.ix_(o, o)] + w * (A @ A.T)
                        Kb = K0b[:, o] + w * (B @ A.T)
                        P[:, j] = my[j] + Kb @ _psolve(K, Yc[o, j])
                res[(a, v, w)] = P
    return res


def _mean_pearson(P, Yobs):
    M = ~np.isnan(Yobs)
    cnt = M.sum(0)
    valid = cnt >= 3
    if not valid.any():
        return 0.0
    c = np.maximum(cnt, 1)
    mp = np.where(M, P, 0.0).sum(0) / c
    my = np.where(M, Yobs, 0.0).sum(0) / c
    dp = np.where(M, P - mp, 0.0)
    dy = np.where(M, Yobs - my, 0.0)
    num = (dp * dy).sum(0)
    sp = np.sqrt((dp ** 2).sum(0))
    sy = np.sqrt((dy ** 2).sum(0))
    ok = (sp > 1e-12) & (sy > 1e-12)
    r = np.where(ok, num / np.where(ok, sp * sy, 1.0), 0.0)
    return float(np.mean(r[valid]))


def _sqdist(A, B):
    sa = (A ** 2).sum(1)
    sb = (B ** 2).sum(1)
    D = sa[:, None] + sb[None, :] - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    gvar = F_train.var().sort_values(ascending=False)
    genes = gvar.index[:5000]
    mu = F_train[genes].values.mean(0)
    sd = F_train[genes].values.std(0)
    sd[sd < 1e-8] = 1.0
    Xtr = (F_train[genes].values - mu) / sd
    Xte = (F_test[genes].values - mu) / sd

    # broader gene pool for the sparse per-drug biomarker model
    sgenes = gvar.index[:SP_GENES]
    Str = F_train[sgenes].values.astype(float)
    Ste = F_test[sgenes].values.astype(float)

    # RBF kernel on standardized genes, bandwidth = median pairwise squared distance (label-free)
    D2tr = _sqdist(Xtr, Xtr)
    D2te = _sqdist(Xte, Xtr)
    iu = np.triu_indices(len(Xtr), k=1)
    med = np.median(D2tr[iu]) if len(iu[0]) else 1.0
    gamma = 1.0 / max(med, 1e-12)
    Rtr = np.exp(-gamma * D2tr)
    Rte = np.exp(-gamma * D2te)

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

    N = _moa_matrix(ctx, drug_ids)
    b_grid = B_GRID if N.any() else [0.0]
    c_grid = C_GRID

    Yobs = y_train.values.astype(float)

    has_t = any(tlist) and Ttr.shape[1] > 0
    w_grid = W_GRID if has_t else [0.0]
    combos = [(a, v, w) for a in A_GRID for v in V_GRID for w in w_grid]

    best_k, best, best_l, best_g, best_b, best_c = None, (ALPHA_DEFAULT, 0.0, 0.0), 0.0, 0, 0.0, 0.0
    groups = np.asarray(ctx.get("groups_train", np.arange(len(F_train))))
    n_splits = min(3, len(np.unique(groups)))
    if n_splits >= 2:
        splits = list(GroupKFold(n_splits=n_splits).split(Xtr, groups=groups))
        scores = {}
        for kk in K_GRID:
            folds = []
            for tr, va in splits:
                Yt = _winsor(Yobs[tr], kk)
                _, my_f, sd_f = _ystats(Yt)
                N2_f = _corr_matrix(Yt)
                res = _predict_all(Xtr[tr], Rtr[np.ix_(tr, tr)], Ttr[tr], Yt,
                                   Xtr[va], Rtr[np.ix_(va, tr)], Ttr[va], tlist, combos)
                Sps = (_sparse_predict(Str[tr], Yt, Str[va]), _sparse_bagged(Str[tr], Yt, Str[va]))
                folds.append((va, res, Sps, my_f, sd_f, N2_f))
            for cb in combos:
                for lam in L_GRID:
                    for g in ([0] if lam == 0 else [0, 1]):
                        for b in b_grid:
                            for c in c_grid:
                                P = np.zeros_like(Yobs)
                                for va, res, Sps, my_f, sd_f, N2_f in folds:
                                    base = res[cb] if lam == 0 else (1.0 - lam) * res[cb] + lam * Sps[g]
                                    P[va] = _share(base, my_f, sd_f, N, b, N2_f, c)
                                scores[(kk, cb, lam, g, b, c)] = _mean_pearson(P, Yobs)
        best_k, best, best_l, best_g, best_b, best_c = max(
            scores.keys(),
            key=lambda k: (scores[k], k[0] is None, -abs(np.log(k[1][0] / ALPHA_DEFAULT)),
                           -k[1][1], -k[1][2], -k[2], -k[3], -k[4], -k[5]))

    Yfit = _winsor(Yobs, best_k)
    res = _predict_all(Xtr, Rtr, Ttr, Yfit, Xte, Rte, Tte, tlist, [best])
    P = res[best]
    if best_l > 0:
        Sp = _sparse_bagged(Str, Yfit, Ste) if best_g == 1 else _sparse_predict(Str, Yfit, Ste)
        P = (1.0 - best_l) * P + best_l * Sp
    _, my_all, sd_all = _ystats(Yfit)
    N2 = _corr_matrix(Yfit) if best_c != 0 else None
    return _share(P, my_all, sd_all, N, best_b, N2, best_c)
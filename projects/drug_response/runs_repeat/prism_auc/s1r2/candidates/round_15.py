"""Shared ridge on top-5,000 variance genes, blended with a per-drug ridge on genes screened by
correlation with that drug's AUC inside the training fold. Screen variants: raw correlations,
correlations smoothed with MOA/target neighbour drugs, and (NEW) a target-neighbourhood screen
that ranks genes only within the drug's annotated target genes plus their STRING (score >= 700)
first neighbours -- a biologically restricted candidate set that suffers far less multiple-testing
noise than a genome-wide screen. Besides hard top-k screens, a soft screen weights each of the
top-1000 genes by (|r| / mean|r|)^p. The (screen, penalty, variant, blend weight) option is chosen
PER DRUG, shrunk toward the global optimum, with the shrinkage chosen by cross-fitting over inner
folds. Prediction-level drug sharing mixes each drug's z-scored prediction with the mean
prediction of its neighbour drugs, with neighbour type and eta chosen by cross-fitting."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_POOL = 10000
N_BASE = 5000
ALPHA_BASE = 5000.0
SCREENS = ((50, 0), (200, 0), (1000, 0), (1000, 1), (1000, 2))  # (k, soft-weight power p)
LAMS = (1.0, 5.0, 25.0)
VARIANTS = ("raw", "moa", "tgt")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRATS = [("all", 1.0)] + [(m, s) for m in ("w", "all") for s in (0.0, 0.25, 0.5, 0.75)]
SHARES = [(None, 0.0)] + [(k, e) for k in ("moa", "corr") for e in (0.15, 0.3, 0.5)]
CORR_THR = 0.5
CORR_TOP = 10
MIN_TSET = 5

_STRING = {}


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


def _string_edges(ctx):
    """STRING edge list (upper-case symbol arrays a, b), cached; None if unavailable."""
    if "edges" in _STRING:
        return _STRING["edges"]
    edges = None
    try:
        d = ctx.get("genesets_dir")
        gs = ctx.get("genesets")
        names = []
        if isinstance(gs, dict):
            for k, v in gs.items():
                names += [x for x in (k, v) if isinstance(x, str)]
        elif isinstance(gs, (list, tuple)):
            names += [x for x in gs if isinstance(x, str)]
        elif isinstance(gs, str):
            names.append(gs)
        if isinstance(d, str) and os.path.isdir(d):
            names += sorted(os.listdir(d))
        paths = []
        for nm in names:
            low = os.path.basename(nm).lower()
            if "string" not in low and "ppi" not in low:
                continue
            for p in (nm, os.path.join(d, nm) if isinstance(d, str) else None):
                if p and os.path.isfile(p) and p not in paths:
                    paths.append(p)
        for p in paths:
            df = None
            for sep in ("\t", ",", r"\s+"):
                try:
                    t = pd.read_csv(p, sep=sep)
                except Exception:
                    continue
                if t.shape[1] >= 2:
                    df = t
                    break
            if df is None:
                continue
            obj = [c for c in df.columns if df[c].dtype == object]
            if len(obj) < 2:
                continue
            num = [c for c in df.columns if c not in obj and np.issubdtype(df[c].dtype, np.number)]
            if num:
                sc = df[num[-1]].values.astype(float)
                thr = 700.0 if np.nanmax(sc) > 1.0 else 0.7
                df = df[sc >= thr]
            a = df[obj[0]].astype(str).str.upper().values
            b = df[obj[1]].astype(str).str.upper().values
            edges = (a, b)
            break
    except Exception:
        edges = None
    _STRING["edges"] = edges
    return edges


def _target_genes(drugs, D, cols, ctx):
    """Per-drug list of column names: annotated target genes plus STRING first neighbours."""
    out = [[] for _ in range(D)]
    if drugs is None or len(drugs) != D or "target" not in drugs.columns:
        return out
    up = {}
    for c in cols:
        up.setdefault(str(c).upper(), c)
    edges = _string_edges(ctx)
    for j in range(D):
        v = drugs["target"].iloc[j]
        T = {t.strip().upper() for t in v.split(",") if t.strip()} if isinstance(v, str) else set()
        if not T:
            continue
        key = frozenset(T)
        if key in _STRING:
            N = _STRING[key]
        else:
            N = set(T)
            if edges is not None:
                a, b = edges
                Tl = list(T)
                N |= set(b[np.isin(a, Tl)]) | set(a[np.isin(b, Tl)])
            _STRING[key] = N
        out[j] = sorted(up[g] for g in N if g in up)
    return out


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


def _ridge(Xk, Xbk, y, lams):
    """Ridge predictions for several penalties; primal or dual eigendecomposition, whichever is smaller."""
    n, k = Xk.shape
    if k < n:
        w, V = np.linalg.eigh(Xk.T @ Xk)
        w = np.clip(w, 0, None)
        Vy = V.T @ (Xk.T @ y)
        Bt = Xbk @ V
    else:
        w, V = np.linalg.eigh(Xk @ Xk.T)
        w = np.clip(w, 0, None)
        Vy = V.T @ y
        Bt = (Xbk @ Xk.T) @ V
    return [Bt @ (Vy / (w + lam)) for lam in lams]


def _models(Xtr_raw, Ytr, Xte_raw, An, has_nb, tsets, n0):
    """Returns ymu, baseline deviations (n_te x D), dict of screened deviations per (screen, lam, variant)."""
    Ztr, Zte = _std(Xtr_raw, Xte_raw)
    ymu = np.nan_to_num(np.nanmean(Ytr, 0))
    Yc = np.where(np.isnan(Ytr), ymu, Ytr) - ymu
    n, D = Yc.shape
    B = Ztr[:, :N_BASE]
    base = (Zte[:, :N_BASE] @ B.T) @ np.linalg.solve(B @ B.T + ALPHA_BASE * np.eye(n), Yc)
    ysd = Yc.std(0)
    ysd[ysd < 1e-12] = 1.0
    C = (Ztr.T @ Yc) / (n * ysd)  # genes x drugs correlations
    absC = np.abs(C)
    Smoa = np.abs((C[:n0] + C[:n0] @ An.T) / 2.0)
    kmax = max(k for k, _ in SCREENS)
    ord_raw = np.argsort(-absC[:n0], axis=0)[:kmax]
    ord_moa = np.argsort(-Smoa, axis=0)[:kmax]
    scr = {(s, l, v): np.zeros((Zte.shape[0], D)) for s in SCREENS for l in LAMS for v in VARIANTS}
    for j in range(D):
        for v in VARIANTS:
            if (v == "moa" and not has_nb[j]) or (v == "tgt" and tsets[j] is None):
                for s in SCREENS:
                    for l in LAMS:
                        scr[(s, l, v)][:, j] = scr[(s, l, "raw")][:, j]
                continue
            if v == "raw":
                idx = ord_raw[:, j]
                sv = absC[idx, j]
            elif v == "moa":
                idx = ord_moa[:, j]
                sv = Smoa[idx, j]
            else:
                T = tsets[j]
                idx = T[np.argsort(-absC[T, j])[:kmax]]
                sv = absC[idx, j]
            Xa, Xb = Ztr[:, idx], Zte[:, idx]
            for (k, p) in SCREENS:
                ke = min(k, len(idx))
                Xk, Xbk = Xa[:, :ke], Xb[:, :ke]
                if p > 0:
                    wv = sv[:ke] / max(sv[:ke].mean(), 1e-12)
                    wv = wv ** p
                    wv = wv / max(wv.mean(), 1e-12)
                    sq = np.sqrt(wv)
                    Xk, Xbk = Xk * sq, Xbk * sq
                preds = _ridge(Xk, Xbk, Yc[:, j], [l * ke for l in LAMS])
                for l, pr in zip(LAMS, preds):
                    scr[((k, p), l, v)][:, j] = pr
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
    variances = F_train.var()
    genes = list(variances.sort_values(ascending=False).index[:N_POOL])
    n0 = len(genes)
    Y = y_train.values.astype(float)
    D = Y.shape[1]
    groups = ctx.get("groups_train")
    groups = np.arange(len(F_train)) if groups is None else np.asarray(groups)

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

    # target-neighbourhood gene sets (label-free: drug annotations + STRING)
    try:
        tg_cols = _target_genes(drugs, D, list(F_train.columns), ctx)
    except Exception:
        tg_cols = [[] for _ in range(D)]
    in_pool = set(genes)
    extras = sorted({g for gl in tg_cols for g in gl if g not in in_pool and variances.get(g, 0.0) > 1e-8})
    cols = genes + extras
    pos = {g: i for i, g in enumerate(cols)}
    tsets = []
    for gl in tg_cols:
        ii = np.array(sorted({pos[g] for g in gl if g in pos}), dtype=int)
        tsets.append(ii if len(ii) >= MIN_TSET else None)

    Xtr = F_train[cols].values.astype(float)
    Xte = F_test[cols].values.astype(float)

    # grouped inner CV out-of-fold predictions for every (screen, lam, variant)
    configs = [(s, l, v) for s in SCREENS for l in LAMS for v in VARIANTS]
    oof_b = np.zeros_like(Y)
    oof_s = {c: np.zeros_like(Y) for c in configs}
    vas = []
    for tr, va in GroupKFold(n_splits=3).split(Xtr, groups=groups):
        vas.append(va)
        _, b, s = _models(Xtr[tr], Y[tr], Xtr[va], An, has_nb, tsets, n0)
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

    ymu, b, s = _models(Xtr, Y, Xte, An, has_nb, tsets, n0)
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
"""Shared ridge on top-5,000 variance genes, blended with a per-drug ridge on genes screened by
correlation with that drug's AUC inside the training fold. Screening variants: raw correlation,
correlation smoothed with the mean correlation vector of the drug's MOA/target neighbours, and
(NEW) correlation propagated one step over the STRING PPI network (each gene's correlation averaged
with the mean correlation of its STRING neighbours), so that genes in co-correlated modules are
preferred over isolated spurious hits. Besides hard top-k screens, a soft screen weights each of
the top-1000 genes by (|r| / mean|r|)^p. The (screen, penalty, variant, blend weight) option is
chosen PER DRUG, shrunk toward the global optimum, with shrinkage chosen by cross-fitting over the
inner grouped folds. Ridge fits with k < n use the (equivalent) primal form for speed."""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold

N_POOL = 10000
N_BASE = 5000
ALPHA_BASE = 5000.0
SCREENS = ((50, 0), (200, 0), (1000, 0), (1000, 1), (1000, 2))  # (k, soft-weight power p)
LAMS = (1.0, 5.0, 25.0)
VARIANTS = ("raw", "moa", "net")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
STRATS = [("all", 1.0)] + [(m, s) for m in ("w", "all") for s in (0.0, 0.25, 0.5, 0.75)]

_NET = {}


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


def _strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            vs = _strings(v)
            out += vs if vs else []
            if isinstance(k, str) and not vs:
                out.append(k)
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out += _strings(v)
        return out
    return []


def _load_string(ctx):
    """STRING edge list (gene1, gene2) as two string arrays, or None. Cached per process."""
    if "edges" in _NET:
        return _NET["edges"]
    edges = None
    try:
        d = ctx.get("genesets_dir")
        d = str(d) if d is not None else ""
        cands = [c for c in _strings(ctx.get("genesets"))
                 if "string" in c.lower() or "ppi" in c.lower()]
        if d and os.path.isdir(d):
            cands += [f for f in sorted(os.listdir(d)) if "string" in f.lower() or "ppi" in f.lower()]
        seen = set()
        for c in cands:
            p = c if os.path.isabs(c) else os.path.join(d, c)
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            for sep in ("\t", ","):
                try:
                    df = pd.read_csv(p, sep=sep)
                except Exception:
                    continue
                obj = [col for col in df.columns if df[col].dtype == object]
                if df.shape[1] >= 2 and len(obj) >= 2:
                    edges = (df[obj[0]].astype(str).values, df[obj[1]].astype(str).values)
                    break
            if edges is not None:
                break
    except Exception:
        edges = None
    _NET["edges"] = edges
    return edges


def _gene_net(genes, ctx):
    """Row-normalized STRING adjacency over the gene pool; isolated genes map to themselves."""
    edges = _load_string(ctx)
    if edges is None:
        return None
    pos = pd.Series(np.arange(len(genes)), index=pd.Index(genes))
    pos = pos[~pos.index.duplicated()]
    a = pd.Series(edges[0]).map(pos).values
    b = pd.Series(edges[1]).map(pos).values
    m = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[m].astype(int), b[m].astype(int)
    keep = a != b
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return None
    G = len(genes)
    A = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(G, G)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    A = sparse.diags(1.0 / np.maximum(deg, 1.0)) @ A + sparse.diags((deg == 0).astype(float))
    return A.tocsr()


def _models(Xtr_raw, Ytr, Xte_raw, An, has_nb, Gn):
    """Returns ymu, baseline deviations (n_te x D), dict of screened deviations per (screen, lam, v)."""
    Ztr, Zte = _std(Xtr_raw, Xte_raw)
    ymu = np.nan_to_num(np.nanmean(Ytr, 0))
    Yc = np.where(np.isnan(Ytr), ymu, Ytr) - ymu
    n, D = Yc.shape
    B = Ztr[:, :N_BASE]
    base = (Zte[:, :N_BASE] @ B.T) @ np.linalg.solve(B @ B.T + ALPHA_BASE * np.eye(n), Yc)
    ysd = Yc.std(0)
    ysd[ysd < 1e-12] = 1.0
    C = (Ztr.T @ Yc) / (n * ysd)  # genes x drugs correlations
    kmax = max(k for k, _ in SCREENS)
    smooth = {"raw": C, "moa": 0.5 * (C + C @ An.T)}
    if Gn is not None:
        smooth["net"] = 0.5 * (C + np.asarray(Gn @ C))
    orders, scores = {}, {}
    for v, M in smooth.items():
        S = np.abs(M)
        orders[v] = np.argsort(-S, axis=0)[:kmax]
        scores[v] = S
    scr = {(s, l, v): np.zeros((Zte.shape[0], D)) for s in SCREENS for l in LAMS for v in VARIANTS}
    for j in range(D):
        for v in VARIANTS:
            if (v == "moa" and not has_nb[j]) or v not in smooth:
                for s in SCREENS:
                    for l in LAMS:
                        scr[(s, l, v)][:, j] = scr[(s, l, "raw")][:, j]
                continue
            idx = orders[v][:, j]
            Xa, Xb = Ztr[:, idx], Zte[:, idx]
            sv = scores[v][idx, j]
            y = Yc[:, j]
            for (k, p) in SCREENS:
                Xk, Xbk = Xa[:, :k], Xb[:, :k]
                if p > 0:
                    wv = sv[:k] / max(sv[:k].mean(), 1e-12)
                    wv = wv ** p
                    wv = wv / wv.mean()
                    sw = np.sqrt(wv)
                    Xk, Xbk = Xk * sw, Xbk * sw
                if k < n:
                    w, V = np.linalg.eigh(Xk.T @ Xk)
                    w = np.clip(w, 0, None)
                    Vy = V.T @ (Xk.T @ y)
                    Pb = Xbk @ V
                    for l in LAMS:
                        scr[((k, p), l, v)][:, j] = Pb @ (Vy / (w + l * k))
                else:
                    Kt = Xbk @ Xk.T
                    w, V = np.linalg.eigh(Xk @ Xk.T)
                    w = np.clip(w, 0, None)
                    Vy = V.T @ y
                    for l in LAMS:
                        scr[((k, p), l, v)][:, j] = Kt @ (V @ (Vy / (w + l * k)))
    return ymu, base, scr


def _select(R, strat, opts):
    """Per-drug option choice from an (n_opt x D) Pearson matrix under a shrinkage strategy."""
    mode, s = strat
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rbar = np.nanmean(R, 1)
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
    try:
        Gn = _gene_net([str(g) for g in genes], ctx)
    except Exception:
        Gn = None

    # grouped inner CV out-of-fold predictions for every (screen, lam, variant)
    configs = [(s, l, v) for s in SCREENS for l in LAMS for v in VARIANTS]
    oof_b = np.zeros_like(Y)
    oof_s = {c: np.zeros_like(Y) for c in configs}
    vas = []
    for tr, va in GroupKFold(n_splits=3).split(Xtr, groups=groups):
        vas.append(va)
        _, b, s = _models(Xtr[tr], Y[tr], Xtr[va], An, has_nb, Gn)
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

    ymu, b, s = _models(Xtr, Y, Xte, An, has_nb, Gn)
    ysd = np.nan_to_num(np.nanstd(Y, 0))
    zb = _zcols(b)
    zs = {}
    blend = np.zeros((Xte.shape[0], D))
    for j in range(D):
        c, wt = opts[choice[j]]
        if c not in zs:
            zs[c] = _zcols(s[c])
        blend[:, j] = (1 - wt) * zb[:, j] + wt * zs[c][:, j]
    return ymu + 0.3 * ysd * blend
"""Top-5,000 variable genes, ridge (alpha=5000) in kernel (dual) form, with a per-drug, less-penalized block of
the drug's annotated target genes (global block weight chosen by grouped inner CV).
Each drug is fit ONLY on the training lines where its AUC was observed (no mean imputation of missing
responses); drugs that share the same observation mask (and kernel) are solved jointly.
Target transformation: models are fit both on raw AUC and on rank-based inverse-normal AUC (rescaled to the
drug's mean/sd); the blend weight between the two is chosen by grouped inner CV (weight 0 = raw-only model).
Expression normalization: the gene kernel is built either from per-gene z-scored log-TPM ("z") or from a
per-gene rank-based inverse-normal transform fit on the training lines (test values mapped into the training
distribution; "r"), which stops dropout zeros / a few extreme lines from dominating line similarity. The
target-gene block always uses z-scored expression (amplification extremes are kept).
Multi-task gene weighting: the line-line kernel is a mix of the uniform gene kernel and a kernel where each
gene is weighted by its mean squared correlation with drug responses (normalized to mean 1). The weights are
either global (all 200 drugs) or drug-cluster specific (KMeans on training response profiles; small clusters
fall back to global weights). Expression version, mix gamma and weighting mode chosen by inner CV."""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]
W_GRID = [0.0, 0.5, 1.0]
_BASE_CFGS = [(0.0, "u"), (0.5, "g"), (1.0, "g"), (0.5, "c"), (1.0, "c")]
G_CFGS = [(e, g, m) for e in ("z", "r") for (g, m) in _BASE_CFGS]
N_CLUST = 8
MIN_CLUST = 5


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


def _rank_gauss_expr(Atr, Ate):
    """Per-gene rank-based inverse-normal transform fit on training lines (mid-ranks for ties); test values
    are placed by their position in the training distribution. Result standardized on training lines."""
    n = Atr.shape[0]
    S = np.sort(Atr, axis=0)

    def tr(A):
        U = np.empty_like(A)
        for g in range(A.shape[1]):
            lo = np.searchsorted(S[:, g], A[:, g], "left")
            hi = np.searchsorted(S[:, g], A[:, g], "right")
            U[:, g] = (0.5 * (lo + hi) + 0.5) / (n + 1)
        return norm.ppf(U)

    Ztr, Zte = tr(Atr), tr(Ate)
    mu = Ztr.mean(0)
    sd = Ztr.std(0)
    sd[sd < 1e-12] = 1.0
    return (Ztr - mu) / sd, (Zte - mu) / sd


def _zscore(Y, obs):
    """Per-drug z-scored observed responses, 0 where unobserved."""
    Yz = np.zeros_like(Y)
    for j in range(Y.shape[1]):
        m = obs[:, j]
        if m.sum() < 3:
            continue
        v = Y[m, j]
        sd = v.std()
        if sd < 1e-12:
            continue
        Yz[m, j] = (v - v.mean()) / sd
    return Yz


def _drug_clusters(Yz, obs):
    """KMeans clustering of drugs by their (normalized) training response profiles; -1 = no/small cluster."""
    D = Yz.shape[1]
    lab = np.full(D, -1)
    ok = obs.sum(0) >= 10
    if ok.sum() < 2 * N_CLUST:
        return lab
    M = Yz[:, ok].T.copy()
    nr = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.maximum(nr, 1e-12)
    l = KMeans(n_clusters=N_CLUST, n_init=10, random_state=0).fit(M).labels_
    sizes = np.bincount(l, minlength=N_CLUST)
    l = np.where(sizes[l] >= MIN_CLUST, l, -1)
    lab[ok] = l
    return lab


def _norm_w(w):
    return w / max(w.mean(), 1e-12)


def _predict_grid(F_train, Ys, F_test, targets, s_grid, g_cfgs):
    """Ys: list of (n_train, D) arrays with NaN for unobserved responses (same mask).
    Returns dict keyed by (e, g, mode, k, s) for kernel config, k-th target version and block weight s."""
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    Atr = F_train[genes].values.astype(float)
    Ate = F_test[genes].values.astype(float)
    mu = Atr.mean(0)
    sd = Atr.std(0)
    sd[sd < 1e-12] = 1.0
    XS = {}
    evers = sorted({e for e, _, _ in g_cfgs})
    if "z" in evers:
        XS["z"] = ((Atr - mu) / sd, (Ate - mu) / sd)
    if "r" in evers:
        XS["r"] = _rank_gauss_expr(Atr, Ate)
    n_te, D = Ate.shape[0], Ys[0].shape[1]
    obs = ~np.isnan(Ys[0])

    need_w = any(g > 0 for _, g, _ in g_cfgs)
    need_c = any(g > 0 and mode == "c" for _, g, mode in g_cfgs)
    labels = np.full(D, -1)
    Yz = None
    if need_w:
        Yz = _zscore(Ys[0], obs)
        if need_c:
            labels = _drug_clusters(Yz, obs)
    nobs = np.maximum(obs.sum(0), 1)

    K0, KW = {}, {}
    for e, (Xtr, Xte) in XS.items():
        K0[e] = (Xtr @ Xtr.T, Xte @ Xtr.T)
        if any(g > 0 and ee == e for ee, g, _ in g_cfgs):
            R2 = ((Xtr.T @ Yz) / nobs) ** 2
            W = {-1: _norm_w(R2.mean(1))}
            if any(g > 0 and mode == "c" and ee == e for ee, g, mode in g_cfgs):
                for c in np.unique(labels):
                    if c >= 0:
                        W[c] = _norm_w(R2[:, labels == c].mean(1))
            for c, w in W.items():
                KW[(e, c)] = ((Xtr * w) @ Xtr.T, (Xte * w) @ Xtr.T)

    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd

    preds = {}
    for e, g, mode in g_cfgs:
        lab = labels if (g > 0 and mode == "c") else np.full(D, -1)
        K0e, Kx0e = K0[e]
        kc = {}
        for c in np.unique(lab):
            if g > 0:
                Kw, Kxw = KW[(e, c)]
                kc[c] = ((1.0 - g) * K0e + g * Kw, (1.0 - g) * Kx0e + g * Kxw)
            else:
                kc[c] = (K0e, Kx0e)
        groups = {}
        for j in range(D):
            if obs[:, j].sum() >= 3:
                groups.setdefault((obs[:, j].tobytes(), lab[j]), []).append(j)
        for k, Y in enumerate(Ys):
            col_mean = np.array([Y[obs[:, j], j].mean() if obs[:, j].any() else 0.0 for j in range(D)])
            # base model (no target block): batch drugs sharing an observation mask and kernel
            base = np.tile(col_mean, (n_te, 1))
            for (_, c), js in groups.items():
                K, Kx = kc[c]
                m = obs[:, js[0]]
                Km = K[np.ix_(m, m)]
                Yc = Y[m][:, js] - col_mean[js]
                base[:, js] = Kx[:, m] @ np.linalg.solve(Km + ALPHA * np.eye(m.sum()), Yc) + col_mean[js]
            for s in s_grid:
                P = base.copy()
                if s > 0 and tg:
                    for j, t in enumerate(targets):
                        t = [x for x in t if x in tset]
                        m = obs[:, j]
                        if not t or m.sum() < 3:
                            continue
                        K, Kx = kc[lab[j]]
                        A = Ttr[t].values[m]
                        B = Tte[t].values
                        Kd = K[np.ix_(m, m)] + (s / len(t)) * (A @ A.T)
                        Kxd = Kx[:, m] + (s / len(t)) * (B @ A.T)
                        yc = Y[m, j] - col_mean[j]
                        P[:, j] = Kxd @ np.linalg.solve(Kd + ALPHA * np.eye(m.sum()), yc) + col_mean[j]
                preds[(e, g, mode, k, s)] = P
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


def _blend(p, gc, w, s):
    e, g, mode = gc
    return (1.0 - w) * p[(e, g, mode, 0, s)] + w * p[(e, g, mode, 1, s)]


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(y_train.columns)
    targets = _drug_targets(ctx, cols, set(F_train.columns))
    Yobs = y_train.values.astype(float)

    # inner grouped CV to choose expression version, kernel config, target-block weight and raw/rank blend
    groups = ctx.get("groups_train")
    idx = np.arange(len(F_train))
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(n_splits=3).split(idx, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(idx)
    combos = [(gi, w, s) for gi in range(len(G_CFGS)) for w in W_GRID for s in S_GRID]
    oof = {c: np.zeros_like(Yobs) for c in combos}
    for tr, te in splits:
        Ytr = Yobs[tr]
        p = _predict_grid(F_train.iloc[tr], [Ytr, _gauss(Ytr)], F_train.iloc[te], targets, S_GRID, G_CFGS)
        for (gi, w, s) in combos:
            oof[(gi, w, s)][te] = _blend(p, G_CFGS[gi], w, s)
    scores = {c: _mean_pearson(oof[c], Yobs) for c in combos}
    best_gi, best_w, best_s = max(combos, key=lambda c: (scores[c], -c[0], -c[1], -c[2]))
    gc = G_CFGS[best_gi]

    p = _predict_grid(F_train, [Yobs, _gauss(Yobs)], F_test, targets, [best_s], [gc])
    return _blend(p, gc, best_w, best_s)
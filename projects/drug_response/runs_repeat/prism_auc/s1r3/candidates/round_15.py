"""Top-5,000 variable genes, standardized, ridge (alpha=5000) in kernel (dual) form, with a per-drug,
less-penalized block of the drug's annotated target genes (global block weight chosen by grouped inner CV).
Each drug is fit ONLY on the training lines where its AUC was observed (no mean imputation of missing
responses); drugs that share the same observation mask (and kernel) are solved jointly.
Target transformation: models are fit both on raw AUC and on rank-based inverse-normal AUC (rescaled to the
drug's mean/sd); the blend weight between the two is chosen by grouped inner CV (weight 0 = raw-only model).
Multi-task gene weighting: the line-line kernel is a mix of the uniform gene kernel and a kernel where each
gene is weighted by its mean squared correlation with drug responses (normalized to mean 1). The weights are
either global (all 200 drugs), drug-cluster specific (KMeans on training response profiles; small clusters
fall back to global weights), or drug-specific: an equal mix of the drug's own squared gene-response
correlation and its cluster weights, so strong single-drug biomarkers get emphasized while the cluster
prior keeps the kernel stable. Mix gamma and weighting mode chosen by inner CV (gamma 0 = uniform kernel)."""
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
G_CFGS = [(0.0, "u"), (0.5, "g"), (1.0, "g"), (0.5, "c"), (1.0, "c"), (0.5, "d"), (1.0, "d")]
N_CLUST = 8
MIN_CLUST = 5
OWN_MIX = 0.5


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
    Returns dict keyed by (g, mode, k, s) for kernel config, k-th target version and block weight s."""
    var = F_train.var()
    genes = var.sort_values(ascending=False).index[:N_GENES]
    mu, sd = F_train[genes].mean(), F_train[genes].std(ddof=0).replace(0, 1.0)
    Xtr = ((F_train[genes] - mu) / sd).values
    Xte = ((F_test[genes] - mu) / sd).values
    n_te, D = Xte.shape[0], Ys[0].shape[1]
    obs = ~np.isnan(Ys[0])

    K0 = Xtr @ Xtr.T
    Kx0 = Xte @ Xtr.T
    labels = np.full(D, -1)
    KW = {}
    W = {}
    R2 = None
    if any(g > 0 for g, _ in g_cfgs):
        Yz = _zscore(Ys[0], obs)
        nobs = np.maximum(obs.sum(0), 1)
        R2 = ((Xtr.T @ Yz) / nobs) ** 2
        W = {-1: _norm_w(R2.mean(1))}
        if any(g > 0 and mode in ("c", "d") for g, mode in g_cfgs):
            labels = _drug_clusters(Yz, obs)
            for c in np.unique(labels):
                if c >= 0:
                    W[c] = _norm_w(R2[:, labels == c].mean(1))
        if any(g > 0 and mode in ("g", "c") for g, mode in g_cfgs):
            for c, w in W.items():
                KW[c] = ((Xtr * w) @ Xtr.T, (Xte * w) @ Xtr.T)

    # standardized target genes (only those with variance in training)
    tg = sorted({g for t in targets for g in t})
    tg = [g for g in tg if var.get(g, 0) > 1e-8]
    tset = set(tg)
    if tg:
        tmu, tsd = F_train[tg].mean(), F_train[tg].std(ddof=0).replace(0, 1.0)
        Ttr = (F_train[tg] - tmu) / tsd
        Tte = (F_test[tg] - tmu) / tsd

    col_means = [np.array([Y[obs[:, j], j].mean() if obs[:, j].any() else 0.0 for j in range(D)])
                 for Y in Ys]

    preds = {}
    for g, mode in g_cfgs:
        if mode == "d" and g > 0:
            continue
        lab = labels if (g > 0 and mode == "c") else np.full(D, -1)
        kc = {}
        for c in np.unique(lab):
            if g > 0:
                Kw, Kxw = KW[c]
                kc[c] = ((1.0 - g) * K0 + g * Kw, (1.0 - g) * Kx0 + g * Kxw)
            else:
                kc[c] = (K0, Kx0)
        groups = {}
        for j in range(D):
            if obs[:, j].sum() >= 3:
                groups.setdefault((obs[:, j].tobytes(), lab[j]), []).append(j)
        for k, Y in enumerate(Ys):
            col_mean = col_means[k]
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
                preds[(g, mode, k, s)] = P

    # drug-specific gene weighting: each drug gets its own kernel (own r^2 mixed with cluster weights)
    d_gs = [g for g, mode in g_cfgs if mode == "d" and g > 0]
    if d_gs:
        nk = len(Ys)
        for g in d_gs:
            for k in range(nk):
                for s in s_grid:
                    preds[(g, "d", k, s)] = np.tile(col_means[k], (n_te, 1))
        for j in range(D):
            m = obs[:, j]
            nm = int(m.sum())
            if nm < 3:
                continue
            wc = W.get(labels[j], W[-1])
            wd = _norm_w((1.0 - OWN_MIX) * wc + OWN_MIX * _norm_w(R2[:, j]))
            Xm = Xtr[m]
            Kd = (Xm * wd) @ Xm.T
            Kxd = (Xte * wd) @ Xm.T
            K0m = K0[np.ix_(m, m)]
            Kx0m = Kx0[:, m]
            cm = np.array([col_means[k][j] for k in range(nk)])
            Yc = np.stack([Ys[k][m, j] for k in range(nk)], axis=1) - cm
            t = [x for x in targets[j] if x in tset]
            if t:
                A = Ttr[t].values[m]
                B = Tte[t].values
                AAt, BAt = A @ A.T, B @ A.T
            I = ALPHA * np.eye(nm)
            for g in d_gs:
                K = (1.0 - g) * K0m + g * Kd
                Kx = (1.0 - g) * Kx0m + g * Kxd
                base = None
                for s in s_grid:
                    if s > 0 and t:
                        Kk = K + (s / len(t)) * AAt
                        Kxk = Kx + (s / len(t)) * BAt
                        out = Kxk @ np.linalg.solve(Kk + I, Yc) + cm
                    else:
                        if base is None:
                            base = Kx @ np.linalg.solve(K + I, Yc) + cm
                        out = base
                    for k in range(nk):
                        preds[(g, "d", k, s)][:, j] = out[:, k]
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
    g, mode = gc
    return (1.0 - w) * p[(g, mode, 0, s)] + w * p[(g, mode, 1, s)]


def fit_predict(F_train, y_train, F_test, ctx):
    cols = list(y_train.columns)
    targets = _drug_targets(ctx, cols, set(F_train.columns))
    Yobs = y_train.values.astype(float)

    # inner grouped CV to choose kernel config, target-block weight and raw/rank blend weight
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
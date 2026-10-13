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
correlation and a prior. The prior is either the drug's KMeans cluster weights ("d") or a soft response-
similarity neighbourhood ("n"): other drugs' squared correlations averaged with weights (positive response
correlation)^4, so close relatives (same MOA / pathway) dominate without a hard cluster boundary.
Mix gamma and weighting mode chosen by inner CV (gamma 0 = uniform kernel).
Post-hoc drug-similarity smoothing: standardized predictions Z_j are replaced by Z_j + lam_j * sum_k S_jk Z_k
with S_jk = (positive training response correlation)^4, pooling the shared signal of close relatives.
A global lam is chosen by grouped inner CV (lam 0 = no smoothing); per-drug lam_j = b/a is then estimated
by non-negative least squares Y_j ~ a Z_j + b (ZS)_j on the inner out-of-fold predictions and shrunk
toward the global lam by rho, chosen by cross-fitting over the inner folds (rho 0 = global lam only)."""
import re
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata, norm
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold

ALPHA = 5000.0
N_GENES = 5000
S_GRID = [0.0, 100.0, 400.0, 1500.0]
W_GRID = [0.0, 0.5, 1.0]
G_CFGS = [(0.0, "u"), (0.5, "g"), (1.0, "g"), (0.5, "c"), (1.0, "c"), (0.5, "d"), (1.0, "d"),
          (0.5, "n"), (1.0, "n")]
L_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
RHO_GRID = [0.0, 0.5, 1.0]
LAM_MAX = 8.0
N_CLUST = 8
MIN_CLUST = 5
OWN_MIX = 0.5
NB_POW = 4


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


def _nb_sim(Yz):
    """Soft drug-drug neighbourhood weights: (positive response correlation)^NB_POW, zero diagonal."""
    C = Yz.T @ Yz
    dg = np.sqrt(np.clip(np.diag(C), 0, None))
    C = C / np.maximum(np.outer(dg, dg), 1e-12)
    S = np.clip(C, 0, None) ** NB_POW
    np.fill_diagonal(S, 0.0)
    S[dg < 1e-12, :] = 0.0
    S[:, dg < 1e-12] = 0.0
    return S


def _smooth_stats(Y):
    obs = ~np.isnan(Y)
    D = Y.shape[1]
    mu = np.zeros(D)
    sd = np.ones(D)
    for j in range(D):
        m = obs[:, j]
        if m.sum() >= 3:
            v = Y[m, j]
            mu[j] = v.mean()
            if v.std() > 1e-12:
                sd[j] = v.std()
    return mu, sd, _nb_sim(_zscore(Y, obs))


def _smooth(P, st, lam):
    lam = np.asarray(lam, dtype=float)
    if np.all(lam <= 0):
        return P
    mu, sd, S = st
    Z = (P - mu) / sd
    return mu + sd * (Z + lam * (Z @ S))


def _drug_lams(Z, N, Yobs, rows, lam0):
    """Per-drug smoothing strength b/a from NNLS fit Y_j ~ a Z_j + b N_j on the given OOF rows."""
    D = Yobs.shape[1]
    lam = np.full(D, float(lam0))
    for j in range(D):
        m = rows & ~np.isnan(Yobs[:, j])
        if m.sum() < 10:
            continue
        y = Yobs[m, j]
        if y.std() < 1e-12:
            continue
        y = (y - y.mean()) / y.std()
        A = np.stack([Z[m, j], N[m, j]], axis=1)
        A = A - A.mean(0)
        if A[:, 1].std() < 1e-12 or A[:, 0].std() < 1e-12:
            continue
        coef, _ = nnls(A, y)
        a, b = coef
        if a <= 1e-12 and b <= 1e-12:
            continue
        lam[j] = min(b / max(a, 1e-12), LAM_MAX)
    return lam


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
    Yz = None
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
        if mode in ("d", "n") and g > 0:
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

    # drug-specific gene weighting: each drug gets its own kernel (own r^2 mixed with a prior:
    # cluster weights for "d", soft response-similarity neighbourhood weights for "n")
    pd_modes = {}
    for g, mode in g_cfgs:
        if mode in ("d", "n") and g > 0:
            pd_modes.setdefault(mode, []).append(g)
    if pd_modes:
        nk = len(Ys)
        WN, nsum = None, None
        if "n" in pd_modes:
            S = _nb_sim(Yz)
            nsum = S.sum(0)
            WN = (R2 @ S) / np.maximum(nsum, 1e-12)
        for mode, gs_ in pd_modes.items():
            for g in gs_:
                for k in range(nk):
                    for s in s_grid:
                        preds[(g, mode, k, s)] = np.tile(col_means[k], (n_te, 1))
        for j in range(D):
            m = obs[:, j]
            nm = int(m.sum())
            if nm < 3:
                continue
            own = _norm_w(R2[:, j])
            Xm = Xtr[m]
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
            for mode, gs_ in pd_modes.items():
                if mode == "d":
                    prior = W.get(labels[j], W[-1])
                else:
                    prior = _norm_w(WN[:, j]) if nsum[j] > 1e-8 else W[-1]
                wd = _norm_w((1.0 - OWN_MIX) * prior + OWN_MIX * own)
                Kd = (Xm * wd) @ Xm.T
                Kxd = (Xte * wd) @ Xm.T
                for g in gs_:
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
                            preds[(g, mode, k, s)][:, j] = out[:, k]
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
    fold_info = []
    for tr, te in splits:
        Ytr = Yobs[tr]
        p = _predict_grid(F_train.iloc[tr], [Ytr, _gauss(Ytr)], F_train.iloc[te], targets, S_GRID, G_CFGS)
        for (gi, w, s) in combos:
            oof[(gi, w, s)][te] = _blend(p, G_CFGS[gi], w, s)
        fold_info.append((te, _smooth_stats(Ytr)))
    scores = {c: _mean_pearson(oof[c], Yobs) for c in combos}
    best_gi, best_w, best_s = max(combos, key=lambda c: (scores[c], -c[0], -c[1], -c[2]))
    gc = G_CFGS[best_gi]

    # choose the global drug-similarity smoothing strength on the same out-of-fold predictions
    base_oof = oof[(best_gi, best_w, best_s)]
    lam_scores = {}
    for lam in L_GRID:
        O = np.zeros_like(Yobs)
        for te, st in fold_info:
            O[te] = _smooth(base_oof[te], st, lam)
        lam_scores[lam] = _mean_pearson(O, Yobs)
    best_lam = max(L_GRID, key=lambda l: (lam_scores[l], -l))

    # per-drug smoothing strength (NNLS own vs neighbour prediction), shrunk toward the global lam;
    # shrinkage rho chosen by cross-fitting over the inner folds
    n = len(Yobs)
    Zo = np.zeros_like(Yobs)
    No = np.zeros_like(Yobs)
    fid = np.zeros(n, dtype=int)
    for f, (te, (mu, sd, S)) in enumerate(fold_info):
        Z = (base_oof[te] - mu) / sd
        Zo[te] = Z
        No[te] = Z @ S
        fid[te] = f
    rho_oof = {r: np.zeros_like(Yobs) for r in RHO_GRID}
    for f, (te, (mu, sd, S)) in enumerate(fold_info):
        lh = _drug_lams(Zo, No, Yobs, fid != f, best_lam)
        for r in RHO_GRID:
            lv = (1.0 - r) * best_lam + r * lh
            rho_oof[r][te] = mu + sd * (Zo[te] + lv * No[te])
    rho_scores = {r: _mean_pearson(rho_oof[r], Yobs) for r in RHO_GRID}
    best_rho = max(RHO_GRID, key=lambda r: (rho_scores[r], -r))
    if best_rho > 0:
        lam_vec = (1.0 - best_rho) * best_lam + best_rho * _drug_lams(Zo, No, Yobs, np.ones(n, bool), best_lam)
    else:
        lam_vec = np.full(Yobs.shape[1], float(best_lam))

    p = _predict_grid(F_train, [Yobs, _gauss(Yobs)], F_test, targets, [best_s], [gc])
    P = _blend(p, gc, best_w, best_s)
    return _smooth(P, _smooth_stats(Yobs), lam_vec)
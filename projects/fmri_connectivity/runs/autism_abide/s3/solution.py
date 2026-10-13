"""Multi-atlas ensemble (cc200, cc400, aal, ho; no GSR): per atlas, label-free DVARS-based motion
scrubbing of high-artifact frames, then tangent-space embedding of Ledoit-Wolf covariances with
per-site Riemannian re-centering (site-wise log-Euclidean reference), followed by label-free per-site
feature standardization (variance shrunk toward the pooled variance) and winsorization of the
standardized features (robustness to artifact-driven outlier values). Large shared modes of the edges
(global connectivity / motion / arousal-like nuisance) are estimated label-free as the top principal
components over all subjects and the top k of them are projected out; k is chosen jointly with an
edge-screening fraction (site-balanced class-mean difference) by repeated, site-stratified grouped
inner CV (within-site AUROC of a mean-difference proxy), which stabilizes the hyperparameter choice.
Finally an L2 logistic regression is trained with site-by-class balanced sample weights; standardized
decision scores are averaged across atlases."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ATLASES = ["cc200", "cc400", "aal", "ho"]
SHRINK_K = 10.0  # pseudo-count pulling each site's feature variance toward the pooled variance
CLIP = 3.0  # winsorization bound (in within-site SD units) for site-standardized features
FRACS = [0.05, 0.1, 0.2, 0.4, 1.0]  # candidate fractions of edges kept after screening
K_OPTS = [0, 1, 2, 4, 8]  # candidate numbers of shared nuisance components removed
INNER_SPLITS = 5
INNER_REPEATS = 3
SCRUB_Z = 3.0  # DVARS outlier threshold (robust SD units above the median)
SCRUB_MAX = 0.2  # never drop more than this fraction of frames
SCRUB_MIN_T = 40  # do not scrub very short scans


def _standardize_ts(x):
    sd = x.std(axis=0)
    dead = sd < 1e-8
    xs = (x - x.mean(axis=0)) / np.where(dead, 1.0, sd)
    xs[:, dead] = 0.0
    return xs, dead


def _scrub(x):
    """Label-free removal of motion-like frames: frames adjacent to a large DVARS jump
    (robust outliers of the RMS frame-to-frame change of standardized ROI signals)."""
    T = x.shape[0]
    if T < SCRUB_MIN_T:
        return x
    xs, dead = _standardize_ts(x)
    live = ~dead
    if live.sum() < 2:
        return x
    d = np.sqrt(np.mean(np.diff(xs[:, live], axis=0) ** 2, axis=1))  # length T-1
    fr = np.zeros(T)
    fr[1:] = d
    fr[:-1] = np.maximum(fr[:-1], d)
    med = np.median(fr)
    mad = 1.4826 * np.median(np.abs(fr - med))
    if mad < 1e-12:
        return x
    bad = fr > med + SCRUB_Z * mad
    cap = int(SCRUB_MAX * T)
    if bad.sum() > cap:
        bad = np.zeros(T, dtype=bool)
        if cap > 0:
            bad[np.argsort(-fr)[:cap]] = True
    if not bad.any():
        return x
    return x[~bad]


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = _scrub(x)
    xs, dead = _standardize_ts(x)
    c = LedoitWolf(assume_centered=True).fit(xs).covariance_
    # dead ROIs: decouple and give unit variance so log() stays finite
    c[dead, :] = 0.0
    c[:, dead] = 0.0
    c[dead, dead] = 1.0
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c


def _fn(C, f):
    w, v = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (v * f(w)) @ v.T


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    atlases = [a for a in ATLASES if a in ctx["atlases"]] or [ctx["atlases"][0]]
    blocks, sizes = [], []
    for atlas in atlases:
        ts = ctx["load_timeseries"](items, atlas, strategy)
        n = ts[0].shape[1]
        iu = np.triu_indices(n, k=0)
        blocks.append(np.stack([_lw_cov(x)[iu] for x in ts]).astype(np.float32))
        sizes.append(n)
    # header: number of atlases, then ROI count per atlas (so fit_predict can unpack)
    header = np.array([len(sizes)] + sizes, dtype=np.float32)
    header = np.tile(header, (len(items), 1))
    return np.concatenate([header] + blocks, axis=1)


def _unpack(F):
    k = int(round(F[0, 0]))
    sizes = [int(round(v)) for v in F[0, 1:1 + k]]
    out, pos = [], 1 + k
    for n in sizes:
        m = n * (n + 1) // 2
        iu = np.triu_indices(n, k=0)
        vec = F[:, pos:pos + m]
        C = np.zeros((F.shape[0], n, n), dtype=np.float64)
        C[:, iu[0], iu[1]] = vec
        C = C + np.transpose(C, (0, 2, 1))
        C[:, np.arange(n), np.arange(n)] *= 0.5
        out.append(C)
        pos += m
    return out


def _sites(meta, n):
    if meta is None:
        return np.array(["_all"] * n)
    try:
        s = meta["site"]
        s = s.values if hasattr(s, "values") else s
        s = np.asarray(s).astype(str)
        if len(s) == n:
            return s
    except Exception:
        pass
    try:
        s = np.asarray([m["site"] for m in meta]).astype(str)
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.array(["_all"] * n)


def _tangent(allC, sites, n_tr):
    n = allC.shape[1]
    logs = np.stack([_fn(C, np.log) for C in allC])
    glob_isqrt = _fn(_fn(logs[:n_tr].mean(axis=0), np.exp), lambda w: 1.0 / np.sqrt(w))
    site_isqrt = {}
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        if len(idx) >= 5:
            ref = _fn(logs[idx].mean(axis=0), np.exp)
            site_isqrt[s] = _fn(ref, lambda w: 1.0 / np.sqrt(w))
        else:
            site_isqrt[s] = glob_isqrt
    iu = np.triu_indices(n, k=0)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X = np.empty((len(allC), len(iu[0])), dtype=np.float64)
    for i, C in enumerate(allC):
        R = site_isqrt[sites[i]]
        M = R @ C @ R
        L = _fn(0.5 * (M + M.T), np.log)
        X[i] = L[iu] * wts
    return X


def _site_standardize(X, sites):
    """Label-free per-site centering and scaling of each feature; site variances are shrunk
    toward the pooled within-site variance so small sites get stable estimates."""
    X = X.copy()
    uniq = np.unique(sites)
    means, vars_, counts = {}, {}, {}
    for s in uniq:
        idx = np.where(sites == s)[0]
        counts[s] = len(idx)
        means[s] = X[idx].mean(axis=0)
        vars_[s] = X[idx].var(axis=0) if len(idx) > 1 else np.zeros(X.shape[1])
    tot = sum(counts.values())
    pooled = sum(counts[s] * vars_[s] for s in uniq) / max(tot, 1)
    pooled = np.maximum(pooled, 1e-12)
    for s in uniq:
        idx = np.where(sites == s)[0]
        ns = counts[s]
        v = (ns * vars_[s] + SHRINK_K * pooled) / (ns + SHRINK_K)
        X[idx] = (X[idx] - means[s]) / np.sqrt(np.maximum(v, 1e-12))
    return X


def _top_pcs(X, kmax):
    """Label-free top principal directions of the rows of X (via the n x n Gram matrix).
    Returns scores T = X V (n x k) and orthonormal loadings V (p x k)."""
    kmax = min(kmax, X.shape[0] - 1)
    if kmax <= 0:
        return np.zeros((X.shape[0], 0)), np.zeros((X.shape[1], 0))
    G = X @ X.T
    w, U = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:kmax]
    w = np.maximum(w[idx], 1e-12)
    V = (X.T @ U[:, idx]) / np.sqrt(w)
    return X @ V, V


def _site_balanced_weights(y, sites):
    """Weights making each site's classes contribute equally (site keeps its total weight);
    single-class sites keep unit weights. Normalized to mean 1."""
    w = np.ones(len(y), dtype=np.float64)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        classes = np.unique(y[idx])
        if len(classes) < 2:
            continue
        for c in classes:
            m = idx[y[idx] == c]
            w[m] = len(idx) / (len(classes) * len(m))
    return w / w.mean()


def _mean_diff(X, pos, w):
    """Weighted positive-minus-negative class mean of each feature."""
    wp, wn = w * pos, w * (~pos)
    return (wp @ X) / max(wp.sum(), 1e-12) - (wn @ X) / max(wn.sum(), 1e-12)


def _within_site_auc(score, pos, sites):
    acc, tot = 0.0, 0
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        if pos[idx].all() or (~pos[idx]).all():
            continue
        acc += len(idx) * roc_auc_score(pos[idx], score[idx])
        tot += len(idx)
    return acc / tot if tot else 0.5


def _inner_splits(groups, sites):
    """Repeated grouped K-fold, stratified by site: groups are shuffled within each site and
    dealt round-robin to folds, so every validation fold contains a share of every site."""
    ug, inv = np.unique(groups, return_inverse=True)
    n_splits = min(INNER_SPLITS, len(ug))
    if n_splits < 2:
        return []
    g_site = np.empty(len(ug), dtype=object)
    g_site[inv] = sites
    _, site_code = np.unique(g_site.astype(str), return_inverse=True)
    out = []
    for r in range(INNER_REPEATS):
        rng = np.random.RandomState(1000 + r)
        order = np.lexsort((rng.rand(len(ug)), site_code))
        offset = rng.randint(n_splits)
        gfold = np.empty(len(ug), dtype=int)
        gfold[order] = (np.arange(len(ug)) + offset) % n_splits
        f = gfold[inv]
        for j in range(n_splits):
            tr, va = np.where(f != j)[0], np.where(f == j)[0]
            if len(tr) and len(va):
                out.append((tr, va))
    return out


def _choose(Xtr, Ttr, V, pos, s_tr, groups):
    """Repeated, site-stratified grouped inner CV jointly over the number of removed nuisance
    components and the screening fraction, using a mean-difference projection as a cheap proxy
    for the heavily regularized logistic regression. Residualized features are never built."""
    k_opts = [k for k in K_OPTS if k <= V.shape[1]]
    combos = [(k, f) for k in k_opts for f in FRACS[::-1]]  # ties -> fewer removed, more kept
    splits = _inner_splits(groups, s_tr)
    if not splits:
        return 0, 1.0
    p = Xtr.shape[1]
    res = np.zeros(len(combos))
    for tr, va in splits:
        w = np.zeros(len(pos))
        w[tr] = _site_balanced_weights(pos[tr].astype(int), s_tr[tr])
        d0 = _mean_diff(Xtr, pos, w)
        dT = _mean_diff(Ttr, pos, w)
        Xv, Tv = Xtr[va], Ttr[va]
        j = 0
        for k in k_opts:
            d = d0 - V[:, :k] @ dT[:k] if k > 0 else d0
            order = np.argsort(-np.abs(d))
            for f in FRACS[::-1]:
                sel = order[:max(1, int(round(f * p)))]
                dm = np.zeros(p)
                dm[sel] = d[sel]
                s = Xv @ dm
                if k > 0:
                    s = s - Tv[:, :k] @ (V[:, :k].T @ dm)
                res[j] += _within_site_auc(s, pos[va], s_tr[va])
                j += 1
    return combos[int(np.argmax(res))]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    n_tr = F_train.shape[0]
    s_tr = _sites(ctx.get("meta_train"), n_tr)
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    sites = np.concatenate([s_tr, s_te])
    y_train = np.asarray(y_train)
    pos = y_train == np.unique(y_train)[-1]
    sw = _site_balanced_weights(y_train, s_tr)
    groups = ctx.get("groups_train")
    groups = np.arange(n_tr) if groups is None else np.asarray(groups)
    if len(groups) != n_tr:
        groups = np.arange(n_tr)
    groups = groups.astype(str)

    covs = _unpack(np.concatenate([F_train, F_test], axis=0))
    score = np.zeros(len(F_test))
    for allC in covs:
        X = _tangent(allC, sites, n_tr)
        X = _site_standardize(X, sites)
        np.clip(X, -CLIP, CLIP, out=X)  # winsorize artifact-driven extreme values
        sc = StandardScaler().fit(X[:n_tr])
        X = sc.transform(X)
        T, V = _top_pcs(X, max(K_OPTS))
        Xtr, Xte = X[:n_tr], X[n_tr:]
        k, frac = _choose(Xtr, T[:n_tr], V, pos, s_tr, groups)
        if k > 0:
            Xtr = Xtr - T[:n_tr, :k] @ V[:, :k].T
            Xte = Xte - T[n_tr:, :k] @ V[:, :k].T
        del X
        if frac < 1.0:
            d = _mean_diff(Xtr, pos, sw)
            sel = np.argsort(-np.abs(d))[:max(1, int(round(frac * len(d))))]
            Xtr, Xte = Xtr[:, sel], Xte[:, sel]
        clf = LogisticRegression(C=0.001, max_iter=5000)
        clf.fit(Xtr, y_train, sample_weight=sw)
        d_tr = clf.decision_function(Xtr)
        d_te = clf.decision_function(Xte)
        score += (d_te - d_tr.mean()) / (d_tr.std() + 1e-12)
    return score / len(covs)
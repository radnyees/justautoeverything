"""Multi-atlas ensemble (cc200, cc400, aal, ho; no GSR): per atlas, tangent-space embedding of
Ledoit-Wolf covariances with per-site Riemannian re-centering (site-wise log-Euclidean reference),
followed by label-free per-site feature standardization (variance shrunk toward the pooled
variance), and L2 logistic regression trained with site-by-class balanced sample weights (so the
model learns within-site case/control contrasts, not site prevalence); standardized decision
scores are averaged across atlases."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ATLASES = ["cc200", "cc400", "aal", "ho"]
SHRINK_K = 10.0  # pseudo-count pulling each site's feature variance toward the pooled variance


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    dead = sd < 1e-8
    xs = (x - x.mean(axis=0)) / np.where(dead, 1.0, sd)
    xs[:, dead] = 0.0
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


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    n_tr = F_train.shape[0]
    s_tr = _sites(ctx.get("meta_train"), n_tr)
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    sites = np.concatenate([s_tr, s_te])
    y_train = np.asarray(y_train)
    sw = _site_balanced_weights(y_train, s_tr)

    covs = _unpack(np.concatenate([F_train, F_test], axis=0))
    score = np.zeros(len(F_test))
    for allC in covs:
        X = _tangent(allC, sites, n_tr)
        X = _site_standardize(X, sites)
        Xtr, Xte = X[:n_tr], X[n_tr:]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(C=0.001, max_iter=5000)
        clf.fit(Xtr, y_train, sample_weight=sw)
        d_tr = clf.decision_function(Xtr)
        d_te = clf.decision_function(Xte)
        score += (d_te - d_tr.mean()) / (d_tr.std() + 1e-12)
    return score / len(covs)
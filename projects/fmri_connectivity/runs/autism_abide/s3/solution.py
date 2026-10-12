"""Multi-atlas ensemble: tangent-space embedding of Ledoit-Wolf shrunk correlation matrices
(CC200, CC400, AAL, HO; no GSR) with per-site Riemannian re-centering (each site's log-Euclidean
mean is used as that site's tangent reference) plus per-site dispersion stretching (Riemannian
Procrustes: each site's tangent vectors scaled to unit RMS norm), isotropic (metric-preserving)
global scaling instead of per-feature standardization, L2 logistic regression per atlas,
averaged standardized scores. Because re-centering removes each site's diagnostic base rate,
an empirical-Bayes site prior (shrunk training prevalence, in log-odds) is added back to the
ensemble score after converting it to an approximate log-odds scale."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

ATLASES = ["cc200", "cc400", "aal", "ho"]
PRIOR_PSEUDO = 20.0   # pseudo-subjects shrinking each site's prevalence toward the global rate
LOGODDS_SLOPE = 1.24  # binormal equal-variance log-odds per unit SD of a score with AUC ~0.78


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps the matrix SPD
    x = x / sd
    cov = LedoitWolf(assume_centered=True).fit(x).covariance_
    R = cov.shape[0]
    cov[np.diag_indices(R)] += 1e-6
    return cov


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    atlases = [a for a in ATLASES if a in ctx["atlases"]]
    blocks, sizes = [], []
    for atlas in atlases:
        ts = ctx["load_timeseries"](items, atlas, strategy)
        R = np.asarray(ts[0]).shape[1]
        iu = np.triu_indices(R, k=0)
        block = np.empty((len(items), len(iu[0])), dtype=np.float32)
        for i, x in enumerate(ts):
            block[i] = _shrunk_corr(x)[iu]
        blocks.append(block)
        sizes.append(R)
        del ts
    # Header: number of atlases, then ROI count per atlas (so fit_predict can unpack)
    header = np.tile(np.asarray([len(sizes)] + sizes, dtype=np.float32), (len(items), 1))
    return np.concatenate([header] + blocks, axis=1)


def _unpack(F):
    n_at = int(round(F[0, 0]))
    sizes = [int(round(v)) for v in F[0, 1:1 + n_at]]
    pos = 1 + n_at
    out = []
    for R in sizes:
        m = R * (R + 1) // 2
        out.append((R, F[:, pos:pos + m]))
        pos += m
    return out


def _to_mat(vec, R, iu):
    C = np.zeros((R, R), dtype=np.float64)
    C[iu] = vec
    C = C + C.T
    C[np.diag_indices(R)] /= 2.0
    return C


def _eig_fn(C, fn):
    w, U = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (U * fn(w)) @ U.T


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        df = pd.DataFrame(meta)
        if "site" in df.columns and len(df) == n:
            return np.asarray(df["site"].astype(str).values, dtype=object)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _recentered_tangent(V_all, R, sites):
    """Tangent vectors where each subject is whitened by its own site's log-Euclidean mean,
    then each site's vectors are stretched to unit RMS norm (equalized site dispersion)."""
    iu = np.triu_indices(R, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    n = len(V_all)
    logs = np.empty((n, len(iu[0])), dtype=np.float64)
    for i, v in enumerate(V_all):
        logs[i] = _eig_fn(_to_mat(v, R, iu), np.log)[iu]
    out = np.empty((n, len(iu[0])), dtype=np.float32)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        logmean = _to_mat(logs[idx].mean(axis=0), R, iu)
        ref = _eig_fn(logmean, np.exp)
        W = _eig_fn(ref, lambda w: 1.0 / np.sqrt(w))
        for i in idx:
            M = W @ _to_mat(V_all[i], R, iu) @ W
            M = (M + M.T) / 2.0
            out[i] = _eig_fn(M, np.log)[iu] * weights
        if len(idx) >= 3:
            disp = np.sqrt(np.mean(np.sum(out[idx].astype(np.float64) ** 2, axis=1)))
            if np.isfinite(disp) and disp > 1e-8:
                out[idx] = out[idx] / disp
    return out


def _atlas_scores(V_train, V_test, R, y_train, sites_train, sites_test):
    n_tr = len(V_train)
    V_all = np.concatenate([V_train, V_test], axis=0)
    sites = np.concatenate([sites_train, sites_test])
    T = _recentered_tangent(V_all, R, sites)
    T_train, T_test = T[:n_tr].astype(np.float64), T[n_tr:].astype(np.float64)

    # Isotropic scaling: center on training mean, one global scale so the mean feature
    # variance is 1 (keeps the Riemannian metric's relative edge weighting intact).
    mu = T_train.mean(axis=0)
    T_train = T_train - mu
    T_test = T_test - mu
    tot_var = np.mean(T_train.var(axis=0))
    g = 1.0 / np.sqrt(tot_var) if np.isfinite(tot_var) and tot_var > 1e-20 else 1.0
    T_train *= g
    T_test *= g

    clf = LogisticRegression(C=0.001 * np.sqrt(19900.0 / T_train.shape[1]), max_iter=5000)
    clf.fit(T_train, y_train)
    d_train = clf.decision_function(T_train)
    d_test = clf.decision_function(T_test)
    s = d_train.std()
    if not np.isfinite(s) or s < 1e-12:
        s = 1.0
    return (d_test - d_train.mean()) / s


def _site_prior(y_pos, sites_train, sites_test):
    """Empirical-Bayes site prevalence (shrunk toward the global rate), as log-odds offset."""
    p0 = np.clip(y_pos.mean(), 1e-3, 1 - 1e-3)
    logit0 = np.log(p0 / (1 - p0))
    prior = np.zeros(len(sites_test), dtype=np.float64)
    for s in np.unique(sites_test):
        m = sites_train == s
        n = m.sum()
        if n == 0:
            continue
        p = (y_pos[m].sum() + PRIOR_PSEUDO * p0) / (n + PRIOR_PSEUDO)
        prior[sites_test == s] = np.log(p / (1 - p)) - logit0
    return prior


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float32)
    F_test = np.asarray(F_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    sites_train = _sites(ctx.get("meta_train"), len(F_train))
    sites_test = _sites(ctx.get("meta_test"), len(F_test))
    tr = _unpack(F_train)
    te = _unpack(F_test)
    score = np.zeros(len(F_test), dtype=np.float64)
    for (R, V_train), (_, V_test) in zip(tr, te):
        score += _atlas_scores(V_train, V_test, R, y_train, sites_train, sites_test)
    score /= max(len(tr), 1)

    # Put the connectivity score on an approximate log-odds scale (unit SD on the test set,
    # label-free), then add back the site base rate that per-site re-centering removed.
    if len(score) >= 5:
        sd = score.std()
        if np.isfinite(sd) and sd > 1e-12:
            score = (score - score.mean()) / sd
    pos_label = np.unique(y_train)[-1]
    y_pos = (y_train == pos_label).astype(np.float64)
    score = LOGODDS_SLOPE * score + _site_prior(y_pos, sites_train, sites_test)
    return 1.0 / (1.0 + np.exp(-score))
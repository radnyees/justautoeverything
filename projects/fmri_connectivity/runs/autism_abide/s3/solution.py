"""Multi-atlas ensemble: tangent-space embedding of Ledoit-Wolf shrunk correlation matrices
(CC200, CC400, AAL, HO; no GSR) with per-site Riemannian re-centering (each site's log-Euclidean
mean is used as that site's tangent reference) plus per-site dispersion stretching (Riemannian
Procrustes: each site's tangent vectors scaled to unit RMS norm), isotropic (metric-preserving)
global scaling instead of per-feature standardization, L2 logistic regression per atlas,
averaged standardized scores. For every atlas (now including CC400) the training set is
augmented with split-half connectivity matrices (first/second half of each scan, separately
re-centered and stretched), which acts as a within-subject-noise-aware regularizer; the same
split-half views are also scored at test time (test-time augmentation, same weights as
training). Training subjects are soft-weighted by head-motion quality (mean framewise
displacement), normalized within class, so motion-corrupted connectomes influence the
classifier less. Because re-centering removes each site's diagnostic base rate, an
empirical-Bayes site prior (shrunk training prevalence, in log-odds) is added back to the
ensemble score after converting it to an approximate log-odds scale."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

ATLASES = ["cc200", "cc400", "aal", "ho"]
PRIOR_PSEUDO = 20.0   # pseudo-subjects shrinking each site's prevalence toward the global rate
LOGODDS_SLOPE = 1.24  # binormal equal-variance log-odds per unit SD of a score with AUC ~0.78
AUG_MAX_ROIS = 450    # split-half augmentation for all atlases used here (incl. CC400)
MIN_HALF_T = 30       # need enough timepoints per half for a usable connectivity estimate
W_FULL, W_HALF = 0.5, 0.25  # per-subject sample weights (sum to 1 per subject)
FD_SCALE = 0.3        # mm; mean FD at which a subject's training weight is halved


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
    blocks, sizes, augs = [], [], []
    for atlas in atlases:
        ts = ctx["load_timeseries"](items, atlas, strategy)
        R = np.asarray(ts[0]).shape[1]
        iu = np.triu_indices(R, k=0)
        aug = R <= AUG_MAX_ROIS
        n_ver = 3 if aug else 1
        vers = [np.empty((len(items), len(iu[0])), dtype=np.float32) for _ in range(n_ver)]
        for i, x in enumerate(ts):
            x = np.asarray(x)
            full = _shrunk_corr(x)[iu]
            vers[0][i] = full
            if aug:
                T = x.shape[0]
                h = T // 2
                if h >= MIN_HALF_T:
                    vers[1][i] = _shrunk_corr(x[:h])[iu]
                    vers[2][i] = _shrunk_corr(x[h:])[iu]
                else:
                    vers[1][i] = full
                    vers[2][i] = full
        blocks.extend(vers)
        sizes.append(R)
        augs.append(1 if aug else 0)
        del ts
    # Header: number of atlases, ROI count per atlas, augmentation flag per atlas
    header = np.tile(np.asarray([len(sizes)] + sizes + augs, dtype=np.float32), (len(items), 1))
    return np.concatenate([header] + blocks, axis=1)


def _unpack(F):
    n_at = int(round(F[0, 0]))
    sizes = [int(round(v)) for v in F[0, 1:1 + n_at]]
    augs = [int(round(v)) for v in F[0, 1 + n_at:1 + 2 * n_at]]
    pos = 1 + 2 * n_at
    out = []
    for R, a in zip(sizes, augs):
        m = R * (R + 1) // 2
        vers = []
        for _ in range(3 if a else 1):
            vers.append(F[:, pos:pos + m])
            pos += m
        out.append((R, vers))
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


def _motion_weights(meta, y_train):
    """Soft quality weights from mean framewise displacement, normalized to mean 1 per class."""
    n = len(y_train)
    w = np.ones(n, dtype=np.float64)
    if meta is None:
        return w
    try:
        df = pd.DataFrame(meta)
        if "mean_fd" not in df.columns or len(df) != n:
            return w
        fd = pd.to_numeric(df["mean_fd"], errors="coerce").values.astype(np.float64)
    except Exception:
        return w
    ok = np.isfinite(fd)
    if ok.sum() < 3:
        return w
    fd = np.where(ok, fd, np.median(fd[ok]))
    fd = np.clip(fd, 0.0, None)
    w = 1.0 / (1.0 + (fd / FD_SCALE) ** 2)
    for c in np.unique(y_train):
        m = y_train == c
        mw = w[m].mean()
        if np.isfinite(mw) and mw > 1e-12:
            w[m] = w[m] / mw
    return w


def _recentered_tangent(V_all, R, sites):
    """Tangent vectors where each subject is whitened by its own site's log-Euclidean mean,
    then each site's vectors are stretched to unit RMS norm (equalized site dispersion)."""
    iu = np.triu_indices(R, k=0)
    weights = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    n = len(V_all)
    out = np.empty((n, len(iu[0])), dtype=np.float32)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        logsum = np.zeros(len(iu[0]), dtype=np.float64)
        for i in idx:
            logsum += _eig_fn(_to_mat(V_all[i], R, iu), np.log)[iu]
        logmean = _to_mat(logsum / len(idx), R, iu)
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


def _iso_scale(T_train, T_test):
    """Center on training mean and apply one global scale (mean feature variance = 1).
    Returned in float32 to keep memory bounded for the large atlases."""
    mu = T_train.mean(axis=0, dtype=np.float64)
    tot_var = np.mean(T_train.var(axis=0, dtype=np.float64))
    g = 1.0 / np.sqrt(tot_var) if np.isfinite(tot_var) and tot_var > 1e-20 else 1.0
    mu32 = mu.astype(np.float32)
    g32 = np.float32(g)
    return (T_train - mu32) * g32, (T_test - mu32) * g32


def _atlas_scores(vers_train, vers_test, R, y_train, sites_train, sites_test, subj_w):
    n_tr = len(vers_train[0])
    sites = np.concatenate([sites_train, sites_test])

    views = []  # (weight, X_train, X_test) per connectivity view
    T = _recentered_tangent(np.concatenate([vers_train[0], vers_test[0]], axis=0), R, sites)
    T_train, T_test = _iso_scale(T[:n_tr], T[n_tr:])
    del T
    if len(vers_train) == 3:
        views.append((W_FULL, T_train, T_test))
        for k in (1, 2):
            Th = _recentered_tangent(np.concatenate([vers_train[k], vers_test[k]], axis=0),
                                     R, sites)
            Th_train, Th_test = _iso_scale(Th[:n_tr], Th[n_tr:])
            del Th
            views.append((W_HALF, Th_train, Th_test))
    else:
        views.append((1.0, T_train, T_test))

    p = T_train.shape[1]
    X = np.empty((n_tr * len(views), p), dtype=np.float64)
    for k, v in enumerate(views):
        X[k * n_tr:(k + 1) * n_tr] = v[1]
    y = np.concatenate([y_train] * len(views))
    sw = np.concatenate([v[0] * subj_w for v in views])

    clf = LogisticRegression(C=0.001 * np.sqrt(19900.0 / p), max_iter=5000)
    clf.fit(X, y, sample_weight=sw)
    # Test-time augmentation: weighted average of decision values over all views.
    d_train = np.zeros(n_tr, dtype=np.float64)
    for k, v in enumerate(views):
        d_train += v[0] * clf.decision_function(X[k * n_tr:(k + 1) * n_tr])
    del X
    d_test = np.zeros(len(vers_test[0]), dtype=np.float64)
    for w, _, Xte in views:
        d_test += w * clf.decision_function(Xte.astype(np.float64))
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
    subj_w = _motion_weights(ctx.get("meta_train"), y_train)
    tr = _unpack(F_train)
    te = _unpack(F_test)
    score = np.zeros(len(F_test), dtype=np.float64)
    for (R, vers_train), (_, vers_test) in zip(tr, te):
        score += _atlas_scores(vers_train, vers_test, R, y_train, sites_train, sites_test,
                               subj_w)
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
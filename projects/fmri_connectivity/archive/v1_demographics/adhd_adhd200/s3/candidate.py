"""Multi-atlas (CC200, AAL, HO) tangent-space LW covariances + relative ROI amplitude (ALFF-like) profile,
each scored by L2 LR trained with site x sex x age-tertile balancing inverse-propensity weights
(hierarchically smoothed; within-stratum signal), stacked with sex/age/site and time-series-derived
motion/quality proxies (relative DVARS level, DVARS spikiness, mean global correlation) via grouped inner CV."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pyriemann.tangentspace import TangentSpace

ATLASES = ["cc200", "aal", "ho"]
N_QC = 3


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    good = sd[0] > 1e-8
    x[:, good] /= sd[:, good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    # dead ROIs (outside FOV): give them unit variance, no coupling
    idx = np.where(~good)[0]
    c[idx, :] = 0.0
    c[:, idx] = 0.0
    c[idx, idx] = 1.0
    c += 1e-3 * np.eye(c.shape[0])
    return c


def _rel_log_amp(x):
    # per-ROI log std (ALFF proxy on band-passed data), centred within subject
    # so that global scanner/site gain cancels; dead ROIs -> subject mean (0)
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    good = sd > 1e-8
    out = np.zeros(x.shape[1])
    if good.any():
        ls = np.log(sd[good])
        out[good] = ls - ls.mean()
    return out


def _qc_proxies(x):
    # label-free motion / data-quality proxies from ROI time series:
    # log median relative DVARS, log(95th pct / median) DVARS spikiness,
    # mean off-diagonal Pearson correlation (motion inflates global coupling)
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    good = sd > 1e-8
    if good.sum() < 2 or x.shape[0] < 3:
        return np.full(N_QC, np.nan)
    z = x[:, good] / sd[good]
    d = np.diff(z, axis=0)
    dv = np.sqrt(np.mean(d ** 2, axis=1))
    med = np.median(dv)
    if not np.isfinite(med) or med <= 0:
        return np.full(N_QC, np.nan)
    spike = np.log(np.percentile(dv, 95) / med)
    k = z.shape[1]
    c = z.T @ z / z.shape[0]
    mcorr = (c.sum() - np.trace(c)) / (k * (k - 1))
    return np.array([np.log(med), spike, mcorr])


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    atlases = [a for a in ATLASES if a in ctx["atlases"]]
    if not atlases:
        atlases = [ctx["atlases"][0]]
    blocks = []
    amp_blocks = []
    header = []
    qc = None
    for atlas in atlases:
        ts = ctx["load_timeseries"](items, atlas, strategy)
        k = ts[0].shape[1]
        iu = np.triu_indices(k)
        blocks.append(np.asarray([_shrunk_corr(x)[iu] for x in ts], dtype=np.float32))
        amp_blocks.append(np.asarray([_rel_log_amp(x) for x in ts], dtype=np.float32))
        if qc is None:
            qc = np.asarray([_qc_proxies(x) for x in ts], dtype=np.float32)
        header.append(k)
    # per-item row: [n_atlases, k_1..k_A, triu(cov_1), ..., triu(cov_A), amp_1, ..., amp_A, qc]
    H = np.tile(np.asarray([len(atlases)] + header, dtype=np.float32), (len(items), 1))
    return np.concatenate([H] + blocks + amp_blocks + [qc], axis=1)


def _unpack(F):
    F = np.asarray(F, dtype=np.float64)
    a = int(round(F[0, 0]))
    ks = [int(round(v)) for v in F[0, 1:1 + a]]
    pos = 1 + a
    covs = []
    for k in ks:
        L = k * (k + 1) // 2
        tri = F[:, pos:pos + L]
        pos += L
        iu = np.triu_indices(k)
        C = np.zeros((F.shape[0], k, k))
        C[:, iu[0], iu[1]] = tri
        C = C + C.transpose(0, 2, 1)
        C[:, np.arange(k), np.arange(k)] *= 0.5
        covs.append(C)
    amp = F[:, pos:pos + sum(ks)]
    pos += sum(ks)
    qc = F[:, pos:pos + N_QC]
    return covs, amp, qc


def _encode_sex(v):
    out = np.full(len(v), np.nan)
    for i, s in enumerate(v):
        if s is None:
            continue
        if isinstance(s, (int, float, np.integer, np.floating)):
            if np.isfinite(s):
                out[i] = float(s)
            continue
        t = str(s).strip().lower()
        if t in ("m", "male", "1", "1.0"):
            out[i] = 1.0
        elif t in ("f", "female", "0", "0.0", "2", "2.0"):
            out[i] = 0.0
    return out


def _covariates(meta, n):
    if meta is None:
        return np.zeros((n, 2))
    meta = pd.DataFrame(meta).reset_index(drop=True)
    sex = _encode_sex(list(meta["sex"])) if "sex" in meta else np.full(n, np.nan)
    age = pd.to_numeric(meta["age"], errors="coerce").to_numpy(dtype=float) if "age" in meta else np.full(n, np.nan)
    return np.column_stack([sex, age])


def _sites(meta, n):
    if meta is None:
        return np.array(["_"] * n)
    meta = pd.DataFrame(meta).reset_index(drop=True)
    if "site" not in meta:
        return np.array(["_"] * n)
    return meta["site"].astype(str).to_numpy()


def _age_bins(sites, age, n_bins=3):
    # age tertiles computed within each site (label-free); missing age -> "na"
    out = np.array(["na"] * len(age), dtype=object)
    for s in np.unique(sites):
        m = (sites == s) & np.isfinite(age)
        if m.sum() < 2 * n_bins:
            out[m] = "0"
            continue
        qs = np.quantile(age[m], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[m] = np.searchsorted(qs, age[m], side="right").astype(str)
    return out


def _stratum_balance_weights(y, sites, sex, age, max_w=5.0, prior=4.0):
    # inverse-propensity weights w.r.t. smoothed P(y=1 | site, sex, age-tertile): each
    # cell looks class-balanced, so site, sex and age signatures carry no label
    # information for the base models. Rates shrink cell -> site x sex -> site.
    y = np.asarray(y).astype(int)
    sex_key = np.array(["na" if not np.isfinite(s) else str(int(s)) for s in sex])
    age_key = _age_bins(sites, age)
    w = np.ones(len(y))
    for s in np.unique(sites):
        ms = sites == s
        p_site = (y[ms].sum() + 1.0) / (ms.sum() + 2.0)
        for g in np.unique(sex_key[ms]):
            mg = ms & (sex_key == g)
            p_sex = (y[mg].sum() + prior * p_site) / (mg.sum() + prior)
            for a in np.unique(age_key[mg]):
                m = mg & (age_key == a)
                p = (y[m].sum() + prior * p_sex) / (m.sum() + prior)
                p = np.clip(p, 0.02, 0.98)
                w[m & (y == 1)] = 0.5 / p
                w[m & (y == 0)] = 0.5 / (1.0 - p)
    w = np.clip(w, 1.0 / max_w, max_w)
    return w / w.mean()


def _lr(C=0.001):
    return LogisticRegression(C=C, max_iter=5000)


def _oof_scores(X_tr, X_te, y_train, groups, n_splits, C, w):
    oof = np.zeros(len(y_train))
    for tr, va in GroupKFold(n_splits=n_splits).split(X_tr, y_train, groups):
        m = _lr(C).fit(X_tr[tr], y_train[tr], sample_weight=w[tr])
        oof[va] = m.decision_function(X_tr[va])
    full = _lr(C).fit(X_tr, y_train, sample_weight=w)
    return oof, full.decision_function(X_te)


def _atlas_scores(C_tr, C_te, y_train, groups, n_splits, w):
    # label-free reference + scaling, fit once on training fold
    ts = TangentSpace(metric="riemann").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr)
    X_te = sc.transform(X_te)
    return _oof_scores(X_tr, X_te, y_train, groups, n_splits, 0.001, w)


def _amp_scores(A_tr, A_te, y_train, groups, n_splits, w):
    sc = StandardScaler().fit(A_tr)
    return _oof_scores(sc.transform(A_tr), sc.transform(A_te), y_train, groups, n_splits, 0.01, w)


def _impute(Z_tr, Z_te):
    Z_tr = Z_tr.copy()
    Z_te = Z_te.copy()
    with np.errstate(all="ignore"):
        mu = np.nanmean(Z_tr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    for j in range(Z_tr.shape[1]):
        Z_tr[~np.isfinite(Z_tr[:, j]), j] = mu[j]
        Z_te[~np.isfinite(Z_te[:, j]), j] = mu[j]
    return Z_tr, Z_te


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    covs_tr, amp_tr, qc_tr = _unpack(F_train)
    covs_te, amp_te, qc_te = _unpack(F_test)
    n_te = covs_te[0].shape[0]

    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))

    site_tr = _sites(ctx.get("meta_train"), len(y_train))
    site_te = _sites(ctx.get("meta_test"), n_te)

    # covariates: sex, age (raw, before imputation)
    Z_tr = _covariates(ctx.get("meta_train"), len(y_train))
    Z_te = _covariates(ctx.get("meta_test"), n_te)

    w = _stratum_balance_weights(y_train, site_tr, Z_tr[:, 0], Z_tr[:, 1])

    oofs, tes = [], []
    for C_tr, C_te in zip(covs_tr, covs_te):
        o, t = _atlas_scores(C_tr, C_te, y_train, groups, n_splits, w)
        oofs.append(o)
        tes.append(t)
    o, t = _amp_scores(amp_tr, amp_te, y_train, groups, n_splits, w)
    oofs.append(o)
    tes.append(t)

    # impute covariates and QC proxies with training means
    Z_tr, Z_te = _impute(Z_tr, Z_te)
    Q_tr, Q_te = _impute(qc_tr, qc_te)

    # site one-hot (sites from training fold; unseen sites -> all zeros)
    levels = sorted(set(site_tr))
    if len(levels) > 1:
        H_tr = np.column_stack([(site_tr == s).astype(float) for s in levels])
        H_te = np.column_stack([(site_te == s).astype(float) for s in levels])
    else:
        H_tr = np.zeros((len(y_train), 0))
        H_te = np.zeros((n_te, 0))

    S_tr = np.column_stack(oofs + [Z_tr, Q_tr, H_tr])
    S_te = np.column_stack(tes + [Z_te, Q_te, H_te])
    s_sc = StandardScaler().fit(S_tr)
    meta_clf = LogisticRegression(C=1.0, max_iter=1000)
    meta_clf.fit(s_sc.transform(S_tr), y_train)
    return meta_clf.predict_proba(s_sc.transform(S_te))[:, 1]
"""Multi-atlas (CC200, AAL, HO) tangent-space LW covariances + relative ROI amplitude (ALFF-like) profile,
each scored by L2 LR, stacked with sex/age/site via grouped inner CV."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pyriemann.tangentspace import TangentSpace

ATLASES = ["cc200", "aal", "ho"]


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


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    atlases = [a for a in ATLASES if a in ctx["atlases"]]
    if not atlases:
        atlases = [ctx["atlases"][0]]
    blocks = []
    amp_blocks = []
    header = []
    for atlas in atlases:
        ts = ctx["load_timeseries"](items, atlas, strategy)
        k = ts[0].shape[1]
        iu = np.triu_indices(k)
        blocks.append(np.asarray([_shrunk_corr(x)[iu] for x in ts], dtype=np.float32))
        amp_blocks.append(np.asarray([_rel_log_amp(x) for x in ts], dtype=np.float32))
        header.append(k)
    # per-item row: [n_atlases, k_1..k_A, triu(cov_1), ..., triu(cov_A), amp_1, ..., amp_A]
    H = np.tile(np.asarray([len(atlases)] + header, dtype=np.float32), (len(items), 1))
    return np.concatenate([H] + blocks + amp_blocks, axis=1)


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
    return covs, amp


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


def _lr(C=0.001):
    return LogisticRegression(C=C, max_iter=5000)


def _oof_scores(X_tr, X_te, y_train, groups, n_splits, C):
    oof = np.zeros(len(y_train))
    for tr, va in GroupKFold(n_splits=n_splits).split(X_tr, y_train, groups):
        m = _lr(C).fit(X_tr[tr], y_train[tr])
        oof[va] = m.decision_function(X_tr[va])
    full = _lr(C).fit(X_tr, y_train)
    return oof, full.decision_function(X_te)


def _atlas_scores(C_tr, C_te, y_train, groups, n_splits):
    # label-free reference + scaling, fit once on training fold
    ts = TangentSpace(metric="riemann").fit(C_tr)
    X_tr = ts.transform(C_tr)
    X_te = ts.transform(C_te)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr)
    X_te = sc.transform(X_te)
    return _oof_scores(X_tr, X_te, y_train, groups, n_splits, 0.001)


def _amp_scores(A_tr, A_te, y_train, groups, n_splits):
    sc = StandardScaler().fit(A_tr)
    return _oof_scores(sc.transform(A_tr), sc.transform(A_te), y_train, groups, n_splits, 0.01)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    covs_tr, amp_tr = _unpack(F_train)
    covs_te, amp_te = _unpack(F_test)
    n_te = covs_te[0].shape[0]

    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))

    oofs, tes = [], []
    for C_tr, C_te in zip(covs_tr, covs_te):
        o, t = _atlas_scores(C_tr, C_te, y_train, groups, n_splits)
        oofs.append(o)
        tes.append(t)
    o, t = _amp_scores(amp_tr, amp_te, y_train, groups, n_splits)
    oofs.append(o)
    tes.append(t)

    # covariates: sex, age (impute with training means, standardize)
    Z_tr = _covariates(ctx.get("meta_train"), len(y_train))
    Z_te = _covariates(ctx.get("meta_test"), n_te)
    mu = np.nanmean(Z_tr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    for j in range(Z_tr.shape[1]):
        Z_tr[np.isnan(Z_tr[:, j]), j] = mu[j]
        Z_te[np.isnan(Z_te[:, j]), j] = mu[j]

    # site one-hot (sites from training fold; unseen sites -> all zeros)
    site_tr = _sites(ctx.get("meta_train"), len(y_train))
    site_te = _sites(ctx.get("meta_test"), n_te)
    levels = sorted(set(site_tr))
    if len(levels) > 1:
        H_tr = np.column_stack([(site_tr == s).astype(float) for s in levels])
        H_te = np.column_stack([(site_te == s).astype(float) for s in levels])
    else:
        H_tr = np.zeros((len(y_train), 0))
        H_te = np.zeros((n_te, 0))

    S_tr = np.column_stack(oofs + [Z_tr, H_tr])
    S_te = np.column_stack(tes + [Z_te, H_te])
    s_sc = StandardScaler().fit(S_tr)
    meta_clf = LogisticRegression(C=1.0, max_iter=1000)
    meta_clf.fit(s_sc.transform(S_tr), y_train)
    return meta_clf.predict_proba(s_sc.transform(S_te))[:, 1]
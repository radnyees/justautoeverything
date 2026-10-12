"""Log relative + log absolute band power per channel (Welch PSD) in nine fine clinical sub-bands
(1-2 Hz polymorphic delta vs 2-4 Hz delta, 4-6/6-8 theta, 8-10/10-13 low/high alpha to capture
slowing of the posterior dominant rhythm, 13-20/20-30 beta, 30-45 gamma) plus epoch-level
variability of log relative band power (std and 90th percentile across clean epochs), plus
side-agnostic hemispheric asymmetry features (|left - right| for homologous 10-20 pairs, maxed over
pairs), plus log-Euclidean tangent-space features of spatial covariance matrices in the SAME nine
sub-bands (projected off the average-reference null space), plus scale-invariant waveform-shape
features (Hjorth mobility/complexity, kurtosis, skewness, normalized steepness), all estimated only
on clean 4-s epochs (within-subject robust-z rejection of high-variance / flat / clipped epochs).
Crop augmentation: every recording yields three feature views (all clean epochs, clean epochs of
the first half, clean epochs of the second half); per-epoch quantities are computed once and
re-aggregated per view. Models train on all views (labels/groups repeated) and test predictions are
averaged over the three views (test-time augmentation).
Model: probability average of standardized L2 logistic regression (C chosen by inner grouped CV),
a regularized LightGBM, and an RBF-kernel SVM (median-heuristic width, precomputed kernel, Platt
calibration on patient-grouped inner out-of-fold decision values)."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb

BANDS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
COV_BANDS = BANDS
EPOCH_SEC = 4.0
Z_THRESH = 4.0
CLIP_UV = 799.0
HOMOLOGOUS = [(("Fp1",), ("Fp2",)), (("F3",), ("F4",)), (("C3",), ("C4",)), (("P3",), ("P4",)),
              (("O1",), ("O2",)), (("F7",), ("F8",)), (("T3", "T7"), ("T4", "T8")),
              (("T5", "P7"), ("T6", "P8"))]


def _pair_indices(channels):
    low = {c.lower(): i for i, c in enumerate(channels)}
    pairs = []
    for left, right in HOMOLOGOUS:
        li = next((low[n.lower()] for n in left if n.lower() in low), None)
        ri = next((low[n.lower()] for n in right if n.lower() in low), None)
        if li is not None and ri is not None:
            pairs.append((li, ri))
    return pairs


def _epochs(x, ep):
    n_ep = x.shape[1] // ep
    if n_ep < 1:
        return x[:, None, :], 1
    return x[:, :n_ep * ep].reshape(x.shape[0], n_ep, ep), n_ep


def _clean_mask(x, fs, sos_bb, ep):
    xb = sosfiltfilt(sos_bb, x, axis=-1)
    xe, n_ep = _epochs(xb, ep)
    lv = np.log(xe.var(axis=-1).sum(axis=0) + 1e-12)            # (n_ep,)
    med = np.median(lv)
    mad = 1.4826 * np.median(np.abs(lv - med)) + 1e-6
    z = (lv - med) / mad
    xr, _ = _epochs(x, ep)
    clipped = (np.abs(xr) >= CLIP_UV).any(axis=(0, 2))
    keep = (np.abs(z) <= Z_THRESH) & ~clipped
    min_keep = max(1, n_ep // 2)
    if keep.sum() < min_keep:                                    # fall back to the least deviant half
        keep = np.zeros(n_ep, dtype=bool)
        keep[np.argsort(np.abs(z))[:min_keep]] = True
    return keep, xb


def _psd_epochs(x, fs, ep):
    xe, _ = _epochs(x, ep)
    nper = min(int(2 * fs), xe.shape[-1])
    f, p_ep = welch(xe, fs=fs, nperseg=nper, axis=-1)            # (ch, n_ep, f)
    return f, p_ep


def _bandpower_view(f, p_all, sel, pairs):
    p_ep = p_all[:, sel, :]
    p = p_ep.mean(axis=1)                                        # average over clean epochs
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(bp / tot)
    ab = np.log(bp)
    # epoch-level variability of log relative band power (intermittent slowing / bursts)
    bp_e = np.stack([p_ep[:, :, (f >= a) & (f < b)].sum(axis=-1) for a, b in BANDS], axis=-1) + 1e-12
    tot_e = p_ep[:, :, (f >= 1) & (f <= 45)].sum(axis=-1, keepdims=True) + 1e-12
    rel_e = np.log(bp_e / tot_e)                                 # (ch, n_sel, bands)
    rel_sd = rel_e.std(axis=1)
    rel_p90 = np.percentile(rel_e, 90, axis=1)
    out = [rel.ravel(), ab.ravel(), rel_sd.ravel(), rel_p90.ravel()]
    # side-agnostic hemispheric asymmetry (focal / lateralized slowing on either side)
    if pairs:
        li = np.array([a for a, _ in pairs])
        ri = np.array([b for _, b in pairs])
        d_ab = np.abs(ab[li] - ab[ri])                           # (pairs, bands)
        d_rel = np.abs(rel[li] - rel[ri])
        d_ep = np.percentile(np.abs(rel_e[li] - rel_e[ri]), 90, axis=1)
        for d in (d_ab, d_rel, d_ep):
            out.append(d.ravel())
            out.append(d.max(axis=0))
    return np.concatenate(out)


def _shape_epochs(xb, ep):
    # scale-invariant waveform-shape features on the 1-45 Hz signal, per channel and epoch
    xe, _ = _epochs(xb, ep)
    xe = xe - xe.mean(axis=-1, keepdims=True)
    d1 = np.diff(xe, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = xe.var(axis=-1) + 1e-12
    v1 = d1.var(axis=-1) + 1e-12
    v2 = d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    sd = np.sqrt(v0)
    zs = xe / sd[..., None]
    kurt = (zs ** 4).mean(axis=-1)                               # non-excess kurtosis (>0)
    skew = (zs ** 3).mean(axis=-1)
    steep = np.percentile(np.abs(d1), 99, axis=-1) / sd          # sharpest slopes relative to amplitude
    return np.stack([np.log(mob), np.log(comp), np.log(kurt), np.log(steep), skew], axis=0)


def _shape_view(S, sel):
    s = S[:, :, sel]
    out = []
    for k in range(4):
        out.append(np.median(s[k], axis=1))
        out.append(np.percentile(s[k], 90, axis=1))
    out.append(np.median(s[4], axis=1))
    out.append(np.median(np.abs(s[4]), axis=1))
    return np.concatenate(out)


def _ref_basis(n):
    # orthonormal basis of the subspace orthogonal to the all-ones vector (average-reference null space)
    q, _ = np.linalg.qr(np.concatenate([np.ones((n, 1)), np.eye(n)[:, :n - 1]], axis=1))
    return q[:, 1:]


def _logm_upper(c):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-10 * max(w.max(), 1e-12))
    L = (v * np.log(w)) @ v.T
    iu = np.triu_indices(L.shape[0])
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[iu] * scale


def _cov_epochs(x, sos_list, U, ep):
    covs = []
    for sos in sos_list:
        xf = sosfiltfilt(sos, x, axis=-1)
        xf = xf - xf.mean(axis=0, keepdims=True)          # enforce average reference
        y = U.T @ xf
        ye, _ = _epochs(y, ep)
        covs.append(np.einsum("ikt,jkt->kij", ye, ye) / ye.shape[2])   # (n_ep, d, d)
    return covs


def _cov_view(covs, sel):
    out = []
    for C in covs:
        c = C[sel].mean(axis=0)
        c = c + 1e-6 * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
        out.append(_logm_upper(c))
    return np.concatenate(out)


def featurize(items, ctx):
    fs = ctx["fs"]
    nyq = fs / 2.0
    ep = int(round(EPOCH_SEC * fs))
    sos_list = [butter(4, [a / nyq, min(b, nyq * 0.95) / nyq], btype="band", output="sos")
                for a, b in COV_BANDS]
    sos_bb = butter(4, [1.0 / nyq, min(45.0, nyq * 0.95) / nyq], btype="band", output="sos")
    U = _ref_basis(len(ctx["channels"]))
    pairs = _pair_indices(list(ctx["channels"]))
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        per_view = [[], [], []]
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            keep, xb = _clean_mask(x, fs, sos_bb, ep)
            n_ep = keep.shape[0]
            idx = np.arange(n_ep)
            h = n_ep // 2
            views = [keep]
            for part in (idx < h, idx >= h):   # crops: clean epochs of each half of the recording
                s = keep & part
                views.append(s if s.sum() >= 2 else keep)
            f, p_all = _psd_epochs(x, fs, ep)
            covs = _cov_epochs(x, sos_list, U, ep)
            S = _shape_epochs(xb, ep)
            for v, sel in enumerate(views):
                per_view[v].append(_bandpower_view(f, p_all, sel, pairs))
                per_view[v].append(_cov_view(covs, sel))
                per_view[v].append(_shape_view(S, sel))
        feats.append(np.stack([np.concatenate(pv) for pv in per_view], axis=0))
    F = np.asarray(feats, dtype=np.float32)                      # (n_items, 3 views, n_feat)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _as_views(F):
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 2:
        F = F[:, None, :]
    return F


def _sqdist(A, B):
    a = (A * A).sum(axis=1)[:, None]
    b = (B * B).sum(axis=1)[None, :]
    D = A @ B.T
    D *= -2.0
    D += a
    D += b
    np.maximum(D, 0.0, out=D)
    return D


def _svm_proba(Xtr, ytr, Xte, g):
    # RBF SVM on clipped standardized features, precomputed kernel (BLAS), median-heuristic width
    A = np.clip(Xtr, -5.0, 5.0)
    B = np.clip(Xte, -5.0, 5.0)
    K = _sqdist(A, A)
    rng = np.random.RandomState(0)
    idx = rng.choice(K.shape[0], size=min(1000, K.shape[0]), replace=False)
    sub = K[np.ix_(idx, idx)]
    med = np.median(sub[np.triu_indices(len(idx), k=1)])
    gamma = 1.0 / max(med, 1e-12)
    K *= -gamma
    np.exp(K, out=K)
    Kte = _sqdist(B, A)
    Kte *= -gamma
    np.exp(Kte, out=Kte)
    C = 1.0
    if g is not None and len(np.unique(g)) >= 3:
        dec = np.zeros(len(ytr))
        for tr, va in GroupKFold(n_splits=3).split(K, ytr, g):
            s = SVC(C=C, kernel="precomputed").fit(K[np.ix_(tr, tr)], ytr[tr])
            dec[va] = s.decision_function(K[np.ix_(va, tr)])
        platt = LogisticRegression(C=1e4, max_iter=1000).fit(dec[:, None], ytr)
    else:
        platt = None
    svm = SVC(C=C, kernel="precomputed").fit(K, ytr)
    d_te = svm.decision_function(Kte)
    if platt is not None:
        return platt.predict_proba(d_te[:, None])[:, 1]
    return 1.0 / (1.0 + np.exp(-d_te))


def fit_predict(F_train, y_train, F_test, ctx):
    Ftr = _as_views(F_train)
    Fte = _as_views(F_test)
    n, V, D = Ftr.shape
    m, Vt, _ = Fte.shape
    y_train = np.asarray(y_train)
    Xtr_raw = Ftr.reshape(n * V, D)
    Xte_raw = Fte.reshape(m * Vt, D)
    ytr = np.repeat(y_train, V)

    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    groups = ctx.get("groups_train")
    g = None
    if groups is not None and len(np.unique(groups)) >= 5:
        g = np.repeat(np.asarray(groups), V)               # crops of a patient stay in one inner fold
        cv = list(GroupKFold(n_splits=5).split(Xtr, ytr, g))
        clf = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], cv=cv, scoring="roc_auc",
                                   max_iter=5000, n_jobs=4).fit(Xtr, ytr)
    else:
        clf = LogisticRegression(C=0.03, max_iter=5000).fit(Xtr, ytr)
    p_lr = clf.predict_proba(Xte)[:, 1].reshape(m, Vt).mean(axis=1)

    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, max_depth=-1,
                             min_child_samples=20 * V, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0, random_state=0,
                             n_jobs=4, verbose=-1)
    gbm.fit(Xtr_raw, ytr)
    p_gbm = gbm.predict_proba(Xte_raw)[:, 1].reshape(m, Vt).mean(axis=1)

    p_svm = _svm_proba(Xtr, ytr, Xte, g).reshape(m, Vt).mean(axis=1)
    return (p_lr + p_gbm + p_svm) / 3.0
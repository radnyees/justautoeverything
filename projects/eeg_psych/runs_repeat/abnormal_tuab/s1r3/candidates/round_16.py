"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) + log total power per channel, plus
band covariances (delta/theta/alpha/beta, trace-normalized, shrunk) projected into the Riemannian
tangent space at the TRAINING-FOLD geometric mean (affine-invariant metric, fit inside fit_predict),
plus per-band Riemannian distances to the class geometric means (normal / abnormal; out-of-fold for
training rows) encoding global deviation from the normal spatial structure.
Both are computed over artifact-screened 10-s epochs: epochs that clip or have outlying broadband
amplitude (robust z > 3) are rejected; the spectrum is the median of epoch PSDs and the covariance
is the mean of epoch covariances.
Crop augmentation: features are also computed on the first and second half of each recording's
clean epochs; these half-recording 'views' are added as extra training rows (same label, same
patient group) so the models learn invariance to within-recording state drift. Test predictions use
the full-recording view.
Model: rank-average of a standardized L2 logistic regression (C by grouped inner CV), a regularized
LightGBM that captures non-additive 'any abnormal sign' structure, and an RBF-kernel SVM (C by grouped
inner CV) capturing smooth multivariate deviation from the normal-EEG feature manifold."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
import lightgbm as lgb

EDGES = np.arange(1, 46, 2)          # 1,3,...,45 -> 22 bins of 2 Hz
N_SPEC = len(EDGES)                  # 22 relative bins + log total power per channel
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
SHRINK = 0.05
EPOCH_SEC = 10.0
CLIP_UV = 790.0
Z_MAX = 3.0
SVM_CS = [0.3, 1.0, 3.0]
N_VIEWS = 3                          # full recording, first half, second half (of clean epochs)


def _epoch_view(x, L):
    n_ep = max(1, x.shape[1] // L)
    L = min(L, x.shape[1])
    return x[:, :n_ep * L].reshape(x.shape[0], n_ep, L).transpose(1, 0, 2)   # (ep, ch, L)


def _good_epochs(x, fs):
    """Boolean mask of artifact-free epochs (no clipping, no outlying broadband amplitude)."""
    sos = butter(4, [1, 45], btype="bandpass", fs=fs, output="sos")
    xb = sosfiltfilt(sos, x, axis=-1)
    E = _epoch_view(xb, int(EPOCH_SEC * fs))
    R = _epoch_view(x, int(EPOCH_SEC * fs))
    amp = np.log(E.std(axis=(1, 2)) + 1e-9)
    peak = np.abs(R).max(axis=(1, 2))
    med = np.median(amp)
    mad = 1.4826 * np.median(np.abs(amp - med)) + 1e-9
    good = ((amp - med) / mad < Z_MAX) & (peak < CLIP_UV)
    n = len(amp)
    if good.sum() < max(1, n // 2):                 # too many rejected: keep quieter half
        good = np.zeros(n, dtype=bool)
        good[np.argsort(amp)[:max(1, (n + 1) // 2)]] = True
    return good


def _epoch_psd(x, fs, good):
    E = _epoch_view(x, int(EPOCH_SEC * fs))[good]
    nper = min(int(2 * fs), E.shape[-1])
    f, p = welch(E, fs=fs, nperseg=nper, axis=-1)    # (ep, ch, f)
    return f, p


def _spectral(f, P, sel):
    p = np.median(P[sel], axis=0)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1)
    rel = np.log(bins / tot[:, None] + 1e-12)
    return np.concatenate([rel, np.log(tot)[:, None]], axis=1).ravel()


def _cov_vec(C):
    n = C.shape[0]
    C = C / (np.trace(C) / n + 1e-12)                       # scale invariance
    C = (1 - SHRINK) * C + SHRINK * np.eye(n)                # handles avg-ref rank deficiency
    return C[np.triu_indices(n)]


def _epoch_covs(x, fs, good):
    """Per-epoch band covariances: list over bands of (ep, ch, ch)."""
    out = []
    x = x - x.mean(axis=1, keepdims=True)
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        E = _epoch_view(xf, int(EPOCH_SEC * fs))[good]
        out.append(np.einsum("ecl,edl->ecd", E, E) / E.shape[2])
    return out


def _covfeats(Cs, sel):
    return np.concatenate([_cov_vec(C[sel].mean(axis=0)) for C in Cs])


def _view_sels(n_good):
    idx = np.arange(n_good)
    halves = [h for h in np.array_split(idx, 2)]
    halves = [h if len(h) > 0 else idx for h in halves]
    return [idx] + halves


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        views = [[] for _ in range(N_VIEWS)]
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            good = _good_epochs(x, fs)
            f, P = _epoch_psd(x, fs, good)
            Cs = _epoch_covs(x, fs, good)
            for v, sel in enumerate(_view_sels(P.shape[0])):
                views[v].append(_spectral(f, P, sel))
                views[v].append(_covfeats(Cs, sel))
        feats.append(np.concatenate([np.concatenate(v) for v in views]))
    F = np.asarray(feats, dtype=np.float64)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _n_channels(D, n_cond):
    per = D // n_cond
    for n in range(1, 257):
        if n * N_SPEC + len(COV_BANDS) * n * (n + 1) // 2 == per:
            return n
    raise ValueError("cannot infer channel count")


def _to_mats(V, n):
    iu = np.triu_indices(n)
    M = np.zeros((V.shape[0], n, n))
    M[:, iu[0], iu[1]] = V
    M = M + M.transpose(0, 2, 1)
    M[:, np.arange(n), np.arange(n)] /= 2.0
    # guard against any numerical non-SPD input
    M = 0.5 * (M + M.transpose(0, 2, 1))
    return M


def _riem_dist(Cs, M):
    """Affine-invariant Riemannian distance of each matrix in Cs to reference M."""
    w, V = np.linalg.eigh(M)
    W = (V * (1.0 / np.sqrt(np.clip(w, 1e-12, None)))) @ V.T
    A = np.einsum("ij,njk,kl->nil", W, Cs, W)
    A = 0.5 * (A + A.transpose(0, 2, 1))
    ev = np.clip(np.linalg.eigvalsh(A), 1e-12, None)
    return np.sqrt((np.log(ev) ** 2).sum(axis=1))


def _class_dists(Ctr_views, y, Cte, splits):
    """Distances to normal / abnormal geometric means (estimated on full-recording views);
    out-of-fold for every training view."""
    Ctr = Ctr_views[0]
    Dtr = [np.zeros((len(Ctr), 2)) for _ in Ctr_views]
    for tr, va in splits:
        for k in (0, 1):
            sel = Ctr[tr][y[tr] == k]
            M = mean_riemann(sel) if len(sel) > 0 else mean_riemann(Ctr[tr])
            for v, Cv in enumerate(Ctr_views):
                Dtr[v][va, k] = _riem_dist(Cv[va], M)
    Dte = np.zeros((len(Cte), 2))
    for k in (0, 1):
        sel = Ctr[y == k]
        M = mean_riemann(sel) if len(sel) > 0 else mean_riemann(Ctr)
        Dte[:, k] = _riem_dist(Cte, M)
    add = lambda D: np.column_stack([D, D[:, 0] - D[:, 1]])
    return [add(D) for D in Dtr], add(Dte)


def _tangent(tr_views, F_test, n_cond, y, splits):
    """Replace raw covariance blocks with tangent vectors at the training geometric mean
    (full-recording views), and append per-band distances to the class geometric means."""
    n = _n_channels(F_test.shape[1], n_cond)
    per = n * N_SPEC + len(COV_BANDS) * n * (n + 1) // 2
    m = n * (n + 1) // 2
    tr_parts = [[] for _ in tr_views]
    te_parts = []
    for c in range(n_cond):
        base = c * per
        for v, V in enumerate(tr_views):
            tr_parts[v].append(V[:, base:base + n * N_SPEC])
        te_parts.append(F_test[:, base:base + n * N_SPEC])
        for b in range(len(COV_BANDS)):
            s = base + n * N_SPEC + b * m
            Ctr_views = [_to_mats(V[:, s:s + m], n) for V in tr_views]
            Cte = _to_mats(F_test[:, s:s + m], n)
            ts = TangentSpace(metric="riemann").fit(Ctr_views[0])
            for v, Cv in enumerate(Ctr_views):
                tr_parts[v].append(ts.transform(Cv))
            te_parts.append(ts.transform(Cte))
            dtr, dte = _class_dists(Ctr_views, y, Cte, splits)
            for v in range(len(tr_views)):
                tr_parts[v].append(dtr[v])
            te_parts.append(dte)
    Xtr = [np.nan_to_num(np.concatenate(p, axis=1)) for p in tr_parts]
    Xte = np.nan_to_num(np.concatenate(te_parts, axis=1))
    return Xtr, Xte


def _rank01(p):
    r = rankdata(p)
    return (r - 1) / max(len(r) - 1, 1)


def _fit_svm(Xsel, ysel, Xfit, yfit, Xte, splits):
    """RBF SVM with C chosen by grouped inner CV (AUROC) on full-recording rows; final fit on the
    augmented rows; returns test decision values."""
    best_c, best_auc = SVM_CS[0], -np.inf
    for C in SVM_CS:
        aucs = []
        for tr, va in splits[:3]:
            if len(np.unique(ysel[va])) < 2:
                continue
            m = SVC(C=C, kernel="rbf", gamma="scale", cache_size=1000).fit(Xsel[tr], ysel[tr])
            aucs.append(roc_auc_score(ysel[va], m.decision_function(Xsel[va])))
        a = np.mean(aucs) if aucs else -np.inf
        if a > best_auc:
            best_c, best_auc = C, a
    m = SVC(C=best_c, kernel="rbf", gamma="scale", cache_size=1000).fit(Xfit, yfit)
    return m.decision_function(Xte)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train).astype(int)
    n_tr = len(y)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(n_tr)
    groups = np.asarray(groups)
    D = F_train.shape[1] // N_VIEWS
    tr_views = [F_train[:, v * D:(v + 1) * D] for v in range(N_VIEWS)]
    te_full = F_test[:, :D]
    splits = list(GroupKFold(n_splits=5).split(tr_views[0], y, groups))
    Xtr_views, F_te = _tangent(tr_views, te_full, len(ctx["conditions"]), y, splits)

    F_aug = np.vstack(Xtr_views)
    y_aug = np.tile(y, N_VIEWS)
    g_aug = np.tile(groups, N_VIEWS)
    splits_aug = list(GroupKFold(n_splits=5).split(F_aug, y_aug, g_aug))

    sc = StandardScaler().fit(F_aug)
    Xaug, Xte = sc.transform(F_aug), sc.transform(F_te)
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=splits_aug,
                               scoring="roc_auc", max_iter=5000, n_jobs=1).fit(Xaug, y_aug)
    p_lr = clf.predict_proba(Xte)[:, 1]

    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15,
                             min_child_samples=60, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0, random_state=0,
                             deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
    gbm.fit(F_aug, y_aug)
    p_gb = gbm.predict_proba(F_te)[:, 1]

    Xaug_c = np.clip(Xaug, -6, 6)
    Xte_c = np.clip(Xte, -6, 6)
    p_svm = _fit_svm(Xaug_c[:n_tr], y, Xaug_c, y_aug, Xte_c, splits)

    return (_rank01(p_lr) + _rank01(p_gb) + _rank01(p_svm)) / 3.0
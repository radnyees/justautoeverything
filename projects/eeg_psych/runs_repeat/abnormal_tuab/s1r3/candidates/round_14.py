"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) + log total power per channel, plus an
'eyes-closed-like' relative spectrum (2-Hz bins, 1-21 Hz) from the clean epochs with the strongest
posterior alpha (state-matched posterior dominant rhythm assessment), plus band covariances
(delta/theta/alpha/beta, trace-normalized, shrunk) projected into the Riemannian tangent space at the
TRAINING-FOLD geometric mean (affine-invariant metric, fit inside fit_predict).
All are computed over artifact-screened 10-s epochs: epochs that clip or have outlying broadband
amplitude (robust z > 3) are rejected; the spectrum is the median of epoch PSDs and the covariance
is the mean of epoch covariances. Model: rank-average of a standardized L2 logistic regression
(C by grouped inner CV), a regularized LightGBM that captures non-additive 'any abnormal sign'
structure, and an RBF-kernel SVM (C by grouped inner CV) capturing smooth multivariate deviation
from the normal-EEG feature manifold."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyriemann.tangentspace import TangentSpace
import lightgbm as lgb

EDGES = np.arange(1, 46, 2)          # 1,3,...,45 -> 22 bins of 2 Hz
N_EC = 10                            # EC-like spectrum bins kept: 1-3, ..., 19-21 Hz
N_SPEC = len(EDGES) + N_EC           # 22 relative bins + log total power + 10 EC-like bins per channel
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
SHRINK = 0.05
EPOCH_SEC = 10.0
CLIP_UV = 790.0
Z_MAX = 3.0
EC_FRAC = 0.25
SVM_CS = [0.3, 1.0, 3.0]


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


def _relspec(p, f):
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1)
    return np.log(bins / tot[:, None] + 1e-12), tot


def _spectral(x, fs, good, post):
    E = _epoch_view(x, int(EPOCH_SEC * fs))[good]
    nper = min(int(2 * fs), E.shape[-1])
    f, P = welch(E, fs=fs, nperseg=nper, axis=-1)    # (ep, ch, f)
    rel, tot = _relspec(np.median(P, axis=0), f)
    # eyes-closed-like state: epochs with the strongest posterior alpha (relative power)
    band = (f >= 1) & (f <= 45)
    alpha = (f >= 7) & (f < 13)
    Pp = P[:, post]
    ar = np.log(Pp[:, :, alpha].sum(-1) / (Pp[:, :, band].sum(-1) + 1e-12) + 1e-12).mean(axis=1)
    k = max(1, int(np.ceil(EC_FRAC * len(ar))))
    sel = np.argsort(-ar, kind="stable")[:k]
    rel_ec, _ = _relspec(np.median(P[sel], axis=0), f)
    return np.concatenate([rel, np.log(tot)[:, None], rel_ec[:, :N_EC]], axis=1)


def _cov_vec(C):
    n = C.shape[0]
    C = C / (np.trace(C) / n + 1e-12)                       # scale invariance
    C = (1 - SHRINK) * C + SHRINK * np.eye(n)                # handles avg-ref rank deficiency
    return C[np.triu_indices(n)]


def _covfeats(x, fs, good):
    out = []
    x = x - x.mean(axis=1, keepdims=True)
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        E = _epoch_view(xf, int(EPOCH_SEC * fs))[good]
        C = np.einsum("ecl,edl->cd", E, E) / (E.shape[0] * E.shape[2])
        out.append(_cov_vec(C))
    return np.concatenate(out)


def _posterior_idx(channels, n_ch):
    names = [str(c).upper().replace("EEG", "").strip() for c in (channels or [])]
    post = [i for i, c in enumerate(names) if c.startswith(("O", "P")) and i < n_ch]
    return np.array(post if post else list(range(n_ch)), dtype=int)


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            post = _posterior_idx(ctx.get("channels"), x.shape[0])
            good = _good_epochs(x, fs)
            f.append(_spectral(x, fs, good, post).ravel())
            f.append(_covfeats(x, fs, good))
        feats.append(np.concatenate(f))
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


def _tangent(F_train, F_test, n_cond):
    """Replace raw covariance blocks with tangent vectors at the training geometric mean."""
    n = _n_channels(F_train.shape[1], n_cond)
    per = n * N_SPEC + len(COV_BANDS) * n * (n + 1) // 2
    m = n * (n + 1) // 2
    tr_parts, te_parts = [], []
    for c in range(n_cond):
        base = c * per
        tr_parts.append(F_train[:, base:base + n * N_SPEC])
        te_parts.append(F_test[:, base:base + n * N_SPEC])
        for b in range(len(COV_BANDS)):
            s = base + n * N_SPEC + b * m
            Ctr = _to_mats(F_train[:, s:s + m], n)
            Cte = _to_mats(F_test[:, s:s + m], n)
            ts = TangentSpace(metric="riemann").fit(Ctr)
            tr_parts.append(ts.transform(Ctr))
            te_parts.append(ts.transform(Cte))
    Xtr = np.nan_to_num(np.concatenate(tr_parts, axis=1))
    Xte = np.nan_to_num(np.concatenate(te_parts, axis=1))
    return Xtr, Xte


def _rank01(p):
    r = rankdata(p)
    return (r - 1) / max(len(r) - 1, 1)


def _fit_svm(Xtr, y, Xte, splits):
    """RBF SVM with C chosen by grouped inner CV (AUROC); returns test decision values."""
    best_c, best_auc = SVM_CS[0], -np.inf
    for C in SVM_CS:
        aucs = []
        for tr, va in splits[:3]:
            if len(np.unique(y[va])) < 2:
                continue
            m = SVC(C=C, kernel="rbf", gamma="scale", cache_size=1000).fit(Xtr[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.decision_function(Xtr[va])))
        a = np.mean(aucs) if aucs else -np.inf
        if a > best_auc:
            best_c, best_auc = C, a
    m = SVC(C=best_c, kernel="rbf", gamma="scale", cache_size=1000).fit(Xtr, y)
    return m.decision_function(Xte)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    F_train, F_test = _tangent(F_train, F_test, len(ctx["conditions"]))
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    y = np.asarray(y_train).astype(int)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    splits = list(GroupKFold(n_splits=5).split(Xtr, y, groups))
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=splits,
                               scoring="roc_auc", max_iter=5000, n_jobs=1).fit(Xtr, y)
    p_lr = clf.predict_proba(Xte)[:, 1]

    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0, random_state=0,
                             deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
    gbm.fit(F_train, y)
    p_gb = gbm.predict_proba(F_test)[:, 1]

    Xtr_c = np.clip(Xtr, -6, 6)
    Xte_c = np.clip(Xte, -6, 6)
    p_svm = _fit_svm(Xtr_c, y, Xte_c, splits)

    return (_rank01(p_lr) + _rank01(p_gb) + _rank01(p_svm)) / 3.0
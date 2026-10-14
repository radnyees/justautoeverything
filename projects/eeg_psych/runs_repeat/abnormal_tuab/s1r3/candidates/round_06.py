"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) + log total power per channel, plus
log-Euclidean tangent-space band covariances (delta/theta/alpha/beta, trace-normalized, shrunk),
plus per-channel waveform-shape features (Hjorth mobility/complexity, kurtosis, skewness) that
target paroxysmal transients (spikes, sharp waves) missed by median spectra.
All are computed over artifact-screened 10-s epochs: epochs that clip or have outlying broadband
amplitude (robust z > 3) are rejected; the spectrum is the median of epoch PSDs and the covariance
is the mean of epoch covariances. Model: rank-average of a standardized L2 logistic regression
(C by grouped inner CV) and a regularized LightGBM that captures non-additive 'any abnormal sign'
structure."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

EDGES = np.arange(1, 46, 2)          # 1,3,...,45 -> 22 bins of 2 Hz
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
SHRINK = 0.05
EPOCH_SEC = 10.0
CLIP_UV = 790.0
Z_MAX = 3.0


def _epoch_view(x, L):
    n_ep = max(1, x.shape[1] // L)
    L = min(L, x.shape[1])
    return x[:, :n_ep * L].reshape(x.shape[0], n_ep, L).transpose(1, 0, 2)   # (ep, ch, L)


def _good_epochs(x, fs):
    """Boolean mask of artifact-free epochs (no clipping, no outlying broadband amplitude),
    plus the 1-45 Hz band-passed signal."""
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
    return good, xb


def _spectral(x, fs, good):
    E = _epoch_view(x, int(EPOCH_SEC * fs))[good]
    nper = min(int(2 * fs), E.shape[-1])
    f, p = welch(E, fs=fs, nperseg=nper, axis=-1)    # (ep, ch, f)
    p = np.median(p, axis=0)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1)
    rel = np.log(bins / tot[:, None] + 1e-12)
    return np.concatenate([rel, np.log(tot)[:, None]], axis=1)


def _waveform(xb, fs, good):
    """Per-channel Hjorth mobility/complexity, kurtosis (median + 90th pct over epochs), skewness."""
    E = _epoch_view(xb, int(EPOCH_SEC * fs))[good]           # (ep, ch, L)
    d1 = np.diff(E, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = E.var(axis=-1) + 1e-12
    v1 = d1.var(axis=-1) + 1e-12
    v2 = d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    z = (E - E.mean(axis=-1, keepdims=True)) / np.sqrt(v0)[..., None]
    kurt = np.log(np.mean(z ** 4, axis=-1) + 1e-6)           # log of 4th standardized moment
    skew = np.mean(z ** 3, axis=-1)
    return np.stack([np.log(np.median(mob, axis=0)),
                     np.log(np.median(comp, axis=0)),
                     np.median(kurt, axis=0),
                     np.percentile(kurt, 90, axis=0),
                     np.median(skew, axis=0)], axis=1)


def _logm_vec(C):
    n = C.shape[0]
    C = C / (np.trace(C) / n + 1e-12)                       # scale invariance
    C = (1 - SHRINK) * C + SHRINK * np.eye(n)                # handles avg-ref rank deficiency
    w, V = np.linalg.eigh(C)
    L = (V * np.log(np.maximum(w, 1e-8))) @ V.T
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[iu] * coef


def _covfeats(x, fs, good):
    out = []
    x = x - x.mean(axis=1, keepdims=True)
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        E = _epoch_view(xf, int(EPOCH_SEC * fs))[good]
        C = np.einsum("ecl,edl->cd", E, E) / (E.shape[0] * E.shape[2])
        out.append(_logm_vec(C))
    return np.concatenate(out)


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
            good, xb = _good_epochs(x, fs)
            f.append(_spectral(x, fs, good).ravel())
            f.append(_covfeats(x, fs, good))
            f.append(_waveform(xb, fs, good).ravel())
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _rank01(p):
    r = rankdata(p)
    return (r - 1) / max(len(r) - 1, 1)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
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

    return 0.5 * _rank01(p_lr) + 0.5 * _rank01(p_gb)
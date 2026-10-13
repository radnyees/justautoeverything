"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) + log total power per channel, plus
log-Euclidean tangent-space band covariances (delta/theta/alpha/beta, trace-normalized, shrunk),
plus unsigned hemispheric asymmetries |L-R| of log relative band power / log total power over
homologous 10-20 pairs (and the max over pairs per band) to encode focal abnormality on either side.
All are computed over artifact-screened 10-s epochs: epochs that clip or have outlying broadband
amplitude (robust z > 3) are rejected; the spectrum is the median of epoch PSDs and the covariance
is the mean of epoch covariances. Model: rank-average of a standardized L2 logistic regression
(C by grouped inner CV) and a regularized LightGBM that captures non-additive 'any abnormal sign'
structure."""
import re
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

EDGES = np.arange(1, 46, 2)          # 1,3,...,45 -> 22 bins of 2 Hz
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
# asymmetry bands as index ranges into the 22 two-Hz bins (bin k covers [1+2k, 3+2k) Hz)
ASYM_BANDS = [(0, 2), (2, 4), (4, 6), (6, 15), (15, 22)]   # 1-5, 5-9, 9-13, 13-31, 31-45 Hz
SHRINK = 0.05
EPOCH_SEC = 10.0
CLIP_UV = 790.0
Z_MAX = 3.0


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


def _homologous_pairs(channels):
    """(left_idx, right_idx) for 10-20 pairs like Fp1/Fp2, T3/T4, O1/O2 (odd = left, odd+1 = right)."""
    lut = {c.strip().lower(): i for i, c in enumerate(channels)}
    pairs = []
    for i, c in enumerate(channels):
        m = re.match(r"^([A-Za-z]+)(\d+)$", c.strip())
        if m and int(m.group(2)) % 2 == 1:
            j = lut.get((m.group(1) + str(int(m.group(2)) + 1)).lower())
            if j is not None:
                pairs.append((i, j))
    return pairs


def _asymmetry(S, pairs):
    """Unsigned left-right differences of log relative band power and log total power."""
    if not pairs:
        return np.zeros(0)
    relp = np.exp(S[:, :len(EDGES) - 1])
    B = np.stack([np.log(relp[:, a:b].sum(axis=1) + 1e-12) for a, b in ASYM_BANDS]
                 + [S[:, -1]], axis=1)                       # (ch, 6)
    L = np.array([p[0] for p in pairs])
    R = np.array([p[1] for p in pairs])
    D = np.abs(B[L] - B[R])                                  # (pairs, 6)
    return np.concatenate([D.ravel(), D.max(axis=0)])


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
    pairs = _homologous_pairs(list(ctx["channels"]))
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            good = _good_epochs(x, fs)
            S = _spectral(x, fs, good)
            f.append(S.ravel())
            f.append(_covfeats(x, fs, good))
            f.append(_asymmetry(S, pairs))
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
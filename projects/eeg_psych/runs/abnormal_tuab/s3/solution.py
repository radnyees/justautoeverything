"""Log relative spectrum (1 Hz bins, 1-45 Hz) per channel plus per-channel log absolute total
power (1-45 Hz, median-averaged Welch), plus Log-Euclidean tangent-space vectors of
band-limited (delta/theta/alpha/beta) trace-normalized spatial covariance matrices (log-domain
mean over 2-s epochs), plus epoch-to-epoch variability of log relative band power and log total
power per channel (captures intermittent slowing / bursts / attenuation), plus scale-free
higher-order waveform-morphology statistics per channel (kurtosis, skewness of signal and
derivative, Hjorth mobility/complexity; robust epoch aggregation) to capture spikes / sharp
waves / asymmetric waveforms invisible to second-order (spectral/covariance) features, plus
sign-free hemispheric asymmetry (|L-R| of log absolute / relative band power over homologous
pairs, and the max over pairs) to capture focal abnormalities on either side, plus nonlinear
complexity per channel (multiscale order-4 permutation entropy, Higuchi fractal dimension;
median and low-percentile over 4-s epochs) capturing pathological signal regularity, plus
bipolar "double-banana" montage features (the montage clinicians read TUH EEGs in): per
derivation log relative power in 9 fine bands, log total power and waveform-morphology stats,
emphasizing local potential gradients of focal slowing and phase-reversing transients.
Model: rank-average ensemble of standardized L2 logistic regression (C by grouped inner CV),
a regularized LightGBM (nonlinear / conjunctive abnormality patterns) and an RBF-kernel SVM on
standardized, outlier-clipped features (C by grouped inner CV; smooth nonlinear decision
surface, complementary to linear and axis-aligned tree models)."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb

FMIN, FMAX = 1, 45
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
VAR_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
ASYM_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BIP_BANDS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
HOMOLOGOUS = [(("FP1",), ("FP2",)), (("F3",), ("F4",)), (("F7",), ("F8",)),
              (("C3",), ("C4",)), (("T3", "T7"), ("T4", "T8")), (("P3",), ("P4",)),
              (("T5", "P7"), ("T6", "P8")), (("O1",), ("O2",))]
# longitudinal bipolar ("double banana") montage
_FP1, _FP2, _F3, _F4, _F7, _F8 = ("FP1",), ("FP2",), ("F3",), ("F4",), ("F7",), ("F8",)
_C3, _C4, _P3, _P4, _O1, _O2 = ("C3",), ("C4",), ("P3",), ("P4",), ("O1",), ("O2",)
_T3, _T4, _T5, _T6 = ("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8")
_FZ, _CZ, _PZ = ("FZ",), ("CZ",), ("PZ",)
BIPOLAR = [(_FP1, _F7), (_F7, _T3), (_T3, _T5), (_T5, _O1),
           (_FP2, _F8), (_F8, _T4), (_T4, _T6), (_T6, _O2),
           (_FP1, _F3), (_F3, _C3), (_C3, _P3), (_P3, _O1),
           (_FP2, _F4), (_F4, _C4), (_C4, _P4), (_P4, _O2),
           (_FZ, _CZ), (_CZ, _PZ)]
EPOCH_SEC = 2.0
SHRINK = 0.05
CPLX_EPOCH_SEC = 4.0
PE_ORDER = 4
PE_DELAYS = (1, 2, 4)
HFD_KMAX = 10
SVM_CLIP = 5.0


def _spectral_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    band = (f >= FMIN) & (f < FMAX)
    tot = p[:, band].sum(axis=1, keepdims=True) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].sum(axis=1) for a in range(FMIN, FMAX)], axis=1)
    rel = np.log(bins / tot + 1e-12).ravel()
    # absolute amplitude: robust (median over segments) total power per channel
    _, pm = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    abs_pow = np.log(pm[:, band].sum(axis=1) + 1e-12)
    return np.concatenate([rel, abs_pow])


def _find(names, aliases):
    return next((names.index(n) for n in aliases if n in names), None)


def _pair_indices(channels):
    names = [str(c).upper() for c in channels]
    pairs = []
    for left, right in HOMOLOGOUS:
        li = _find(names, left)
        ri = _find(names, right)
        if li is not None and ri is not None:
            pairs.append((li, ri))
    return pairs


def _bipolar_matrix(channels):
    """Rows of the bipolar derivation matrix; missing electrodes -> zero row (constant feature)."""
    names = [str(c).upper() for c in channels]
    B = np.zeros((len(BIPOLAR), len(names)))
    for k, (a, b) in enumerate(BIPOLAR):
        ai, bi = _find(names, a), _find(names, b)
        if ai is not None and bi is not None:
            B[k, ai] = 1.0
            B[k, bi] = -1.0
    return B


def _bipolar_feats(x, fs, B):
    """Bipolar-montage local features: log relative power in fine bands, log total power
    (median Welch) and scale-free waveform morphology per derivation."""
    xb = B @ x
    f, pm = welch(xb, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = pm[:, (f >= FMIN) & (f < FMAX)].sum(axis=1) + 1e-12
    bp = np.stack([pm[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in BIP_BANDS], axis=1)
    lrel = np.log(bp / tot[:, None] + 1e-12)
    return np.concatenate([lrel.ravel(), np.log(tot), _waveform_feats(xb, fs)])


def _asym_feats(x, fs, pairs):
    """Sign-free hemispheric asymmetry: |log P_L - log P_R| of absolute and relative band power
    (median Welch) per homologous pair, plus max over pairs per band/type."""
    nb = len(ASYM_BANDS)
    if not pairs:
        return np.zeros(2 * nb)
    f, pm = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = pm[:, (f >= FMIN) & (f < FMAX)].sum(axis=1) + 1e-12
    bp = np.stack([pm[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in ASYM_BANDS], axis=1)
    la = np.log(bp + 1e-12)                       # (ch, nb)
    lr = la - np.log(tot)[:, None]
    L = np.array([p[0] for p in pairs])
    R = np.array([p[1] for p in pairs])
    da = np.abs(la[L] - la[R])                    # (pairs, nb)
    dr = np.abs(lr[L] - lr[R])
    return np.concatenate([da.ravel(), dr.ravel(), da.max(axis=0), dr.max(axis=0)])


def _cov_feats(x, fs):
    """Log-Euclidean mean of trace-normalized, shrunk epoch covariances per band -> upper tri."""
    n_ch = x.shape[0]
    iu = np.triu_indices(n_ch)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    L = int(EPOCH_SEC * fs)
    out = []
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        n_ep = xf.shape[1] // L
        if n_ep < 1:
            out.append(np.zeros(len(iu[0])))
            continue
        ep = xf[:, :n_ep * L].reshape(n_ch, n_ep, L).transpose(1, 0, 2)
        ep = ep - ep.mean(axis=-1, keepdims=True)
        C = np.einsum("ect,edt->ecd", ep, ep) / L
        tr = np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-12
        C = C / tr * n_ch                                  # scale-free spatial structure
        C = (1 - SHRINK) * C + SHRINK * np.eye(n_ch)[None]  # handle avg-ref rank deficiency
        evals, evecs = np.linalg.eigh(C)
        logC = np.einsum("eij,ej,ekj->eik", evecs, np.log(np.maximum(evals, 1e-10)), evecs)
        M = logC.mean(axis=0)                              # Log-Euclidean mean (artifact-robust)
        out.append(M[iu] * w)
    return np.concatenate(out)


def _temporal_var_feats(x, fs):
    """Epoch-to-epoch variability of log relative band power and log total power per channel."""
    n_ch = x.shape[0]
    nb = len(VAR_BANDS)
    L = int(EPOCH_SEC * fs)
    n_ep = x.shape[1] // L
    if n_ep < 3:
        return np.zeros(n_ch * (2 * nb + 2))
    ep = x[:, :n_ep * L].reshape(n_ch, n_ep, L)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    P = np.abs(np.fft.rfft(ep * np.hanning(L), axis=-1)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / fs)
    tot = P[..., (f >= FMIN) & (f < FMAX)].sum(axis=-1) + 1e-12          # (ch, ep)
    bp = np.stack([P[..., (f >= lo) & (f < hi)].sum(axis=-1) for lo, hi in VAR_BANDS], axis=-1)
    logrel = np.log(bp / tot[..., None] + 1e-12)                         # (ch, ep, nb)
    logtot = np.log(tot)
    rel_std = logrel.std(axis=1)
    q90, q50 = np.percentile(logrel, [90, 50], axis=1)
    tot_std = logtot.std(axis=1)
    t90, t10 = np.percentile(logtot, [90, 10], axis=1)
    return np.concatenate([rel_std.ravel(), (q90 - q50).ravel(), tot_std, t90 - t10])


def _waveform_feats(x, fs):
    """Scale-free higher-order waveform statistics per channel over 2-s epochs of 1-45 Hz signal:
    log kurtosis (median, p90), skewness of signal and of its derivative (median),
    log Hjorth mobility and complexity (median)."""
    n_ch = x.shape[0]
    L = int(EPOCH_SEC * fs)
    n_ep = x.shape[1] // L
    if n_ep < 3:
        return np.zeros(n_ch * 6)
    sos = butter(4, [FMIN, FMAX], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    ep = xf[:, :n_ep * L].reshape(n_ch, n_ep, L)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    sd0 = ep.std(axis=-1) + 1e-8
    z = ep / sd0[..., None]
    lkurt = np.log(np.mean(z ** 4, axis=-1) + 1e-8)                  # spikiness / transients
    skew = np.mean(z ** 3, axis=-1)                                   # polarity asymmetry
    d1 = np.diff(ep, axis=-1)
    d1 = d1 - d1.mean(axis=-1, keepdims=True)
    sd1 = d1.std(axis=-1) + 1e-8
    dskew = np.mean((d1 / sd1[..., None]) ** 3, axis=-1)             # rise/fall asymmetry (sharp waves)
    d2 = np.diff(d1, axis=-1)
    sd2 = d2.std(axis=-1) + 1e-8
    mob = sd1 / sd0
    comp = (sd2 / sd1) / (mob + 1e-8)
    return np.concatenate([
        np.median(lkurt, axis=1), np.percentile(lkurt, 90, axis=1),
        np.median(skew, axis=1), np.median(dskew, axis=1),
        np.log(np.median(mob, axis=1) + 1e-8), np.log(np.median(comp, axis=1) + 1e-8),
    ])


def _complexity_feats(x, fs):
    """Nonlinear complexity per channel over 4-s epochs of 1-45 Hz signal: multiscale order-4
    permutation entropy (delays 1, 2, 4 samples; median, plus p10 at delay 2) and Higuchi
    fractal dimension (median, p10). Both are invariant to amplitude scale and offset."""
    n_ch = x.shape[0]
    L = int(CPLX_EPOCH_SEC * fs)
    n_ep = x.shape[1] // L
    if n_ep < 3:
        return np.zeros(n_ch * 6)
    sos = butter(4, [FMIN, FMAX], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    ep = xf[:, :n_ep * L].reshape(n_ch * n_ep, L)
    rows = ep.shape[0]
    n_codes = PE_ORDER ** PE_ORDER
    weights = PE_ORDER ** np.arange(PE_ORDER)
    norm = np.log(float(np.prod(np.arange(1, PE_ORDER + 1))))
    pes = []
    for d in PE_DELAYS:
        idx = np.arange(L - (PE_ORDER - 1) * d)[:, None] + d * np.arange(PE_ORDER)[None, :]
        perm = np.argsort(ep[:, idx], axis=-1, kind="stable")
        code = (perm * weights).sum(axis=-1)                         # (rows, n)
        cnt = np.bincount((np.arange(rows)[:, None] * n_codes + code).ravel(),
                          minlength=rows * n_codes).reshape(rows, n_codes).astype(np.float64)
        p = cnt / cnt.sum(axis=1, keepdims=True)
        H = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1) / norm
        pes.append(H.reshape(n_ch, n_ep))
    # Higuchi fractal dimension
    logL = []
    for k in range(1, HFD_KMAX + 1):
        Lk = np.zeros(rows)
        for m in range(k):
            seg = ep[:, m::k]
            nseg = seg.shape[1] - 1
            if nseg < 1:
                continue
            Lk += np.abs(np.diff(seg, axis=1)).sum(axis=1) * (L - 1) / (nseg * k) / k
        logL.append(np.log(Lk / k + 1e-12))
    logL = np.stack(logL, axis=1)                                     # (rows, kmax)
    lk = np.log(np.arange(1, HFD_KMAX + 1, dtype=np.float64))
    lk = lk - lk.mean()
    slope = ((logL - logL.mean(axis=1, keepdims=True)) * lk).sum(axis=1) / (lk ** 2).sum()
    hfd = (-slope).reshape(n_ch, n_ep)
    return np.concatenate([
        np.median(pes[0], axis=1), np.median(pes[1], axis=1), np.median(pes[2], axis=1),
        np.percentile(pes[1], 10, axis=1),
        np.median(hfd, axis=1), np.percentile(hfd, 10, axis=1),
    ])


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    pairs = _pair_indices(ctx["channels"])
    B = _bipolar_matrix(ctx["channels"])
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_spectral_feats(x, fs))
            f.append(_cov_feats(x, fs))
            f.append(_temporal_var_feats(x, fs))
            f.append(_waveform_feats(x, fs))
            f.append(_asym_feats(x, fs, pairs))
            f.append(_complexity_feats(x, fs))
            f.append(_bipolar_feats(x, fs, B))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _rank01(p):
    return (rankdata(p) - 0.5) / len(p)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train).astype(int)
    groups = np.asarray(ctx["groups_train"])
    cv = list(GroupKFold(n_splits=5).split(F_train, y_train, groups))
    F_train = np.nan_to_num(np.asarray(F_train, dtype=np.float64))
    F_test = np.nan_to_num(np.asarray(F_test, dtype=np.float64))

    # linear model
    sc = StandardScaler().fit(F_train)
    Xtr = sc.transform(F_train)
    Xte = sc.transform(F_test)
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               penalty="l2", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    p_lr = clf.predict_proba(Xte)[:, 1]

    # gradient boosting (fixed conservative hyperparameters; nonlinear interactions)
    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.2, reg_lambda=1.0, random_state=0,
                             deterministic=True, force_col_wise=True, n_jobs=-1, verbose=-1)
    gbm.fit(F_train, y_train)
    p_gb = gbm.predict_proba(F_test)[:, 1]

    # RBF-kernel SVM on standardized, outlier-clipped features (C by grouped inner CV)
    Str = np.clip(Xtr, -SVM_CLIP, SVM_CLIP)
    Ste = np.clip(Xte, -SVM_CLIP, SVM_CLIP)
    svm = GridSearchCV(SVC(kernel="rbf", gamma="scale", cache_size=1000),
                       {"C": [0.3, 1.0, 3.0, 10.0]}, cv=cv, scoring="roc_auc",
                       n_jobs=-1).fit(Str, y_train)
    p_sv = svm.decision_function(Ste)

    return (_rank01(p_lr) + _rank01(p_gb) + _rank01(p_sv)) / 3.0
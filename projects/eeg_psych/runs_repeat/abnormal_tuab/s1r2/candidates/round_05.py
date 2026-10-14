"""Log relative power spectrum in 1-Hz bins (1-45 Hz) per channel plus log absolute power in
5 classical bands per channel, from a median-averaged (artifact-robust) Welch PSD, plus
log-Euclidean tangent-space band covariance matrices (delta/theta/alpha/beta, robust epoch
averaging), plus temporal variability (quantile spreads) of 2-s epoch band power capturing
intermittent slowing / paroxysmal events, standardized, L2 logistic regression with C chosen by
inner grouped CV."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BINS = [(k, k + 1) for k in range(1, 45)]
ABS_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
CS = [0.001, 0.003, 0.01, 0.03, 0.1]


def _spectrum(x, fs):
    # median across 2-s segments: transient artifacts (blinks, muscle, electrode pops) no longer
    # dominate the spectrum, leaving the background rhythm that defines (ab)normality
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    df = f[1] - f[0]
    tot = p[:, (f >= 1) & (f < 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BINS], axis=1) / tot + 1e-12)
    # absolute band power (uV^2): relative spectra discard amplitude, yet high-amplitude
    # (focal) delta/theta slowing is a core marker of abnormal clinical EEG
    ab = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) * df for a, b in ABS_BANDS], axis=1) + 1e-6)
    return np.concatenate([rel.ravel(), ab.ravel()])


def _variability(x, fs):
    # the median spectrum suppresses transients by design, but intermittent (focal) slowing and
    # paroxysmal bursts are themselves abnormal findings: describe the per-channel distribution
    # of 2-s epoch power by robust upper / lower quantile spreads
    n = int(2 * fs)
    if x.shape[1] < 2 * n:
        return np.zeros(x.shape[0] * (len(ABS_BANDS) + 1) * 2)
    f, _, S = spectrogram(x, fs=fs, nperseg=n, noverlap=0, axis=-1)   # (ch, f, epochs)
    tot = S[:, (f >= 1) & (f < 45), :].sum(axis=1) + 1e-12            # (ch, epochs)
    bp = np.stack([S[:, (f >= a) & (f < b), :].sum(axis=1) for a, b in ABS_BANDS], axis=1)
    lr = np.log(bp / tot[:, None, :] + 1e-12)                         # (ch, bands, epochs)
    lt = np.log(tot + 1e-12)[:, None, :]
    L = np.concatenate([lr, lt], axis=1)                              # (ch, bands+1, epochs)
    q10, q50, q90 = np.percentile(L, [10, 50, 90], axis=-1)
    return np.concatenate([(q90 - q50).ravel(), (q50 - q10).ravel()])


def _logm_upper(C):
    w, V = np.linalg.eigh(C)
    L = (V * np.log(np.maximum(w, 1e-10))) @ V.T
    n = L.shape[0]
    iu = np.triu_indices(n)
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[iu] * scale


def _covariances(x, fs):
    # spatial covariance per band: captures focal / lateralized slowing and altered spatial
    # coherence that per-channel spectra cannot see; mapped to log-Euclidean tangent space
    n_ch, n_t = x.shape
    ep = int(2 * fs)
    n_ep = max(n_t // ep, 1)
    out = []
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        if n_t >= ep:
            E = xf[:, : n_ep * ep].reshape(n_ch, n_ep, ep).transpose(1, 0, 2)
        else:
            E = xf[None]
        E = E - E.mean(axis=-1, keepdims=True)
        covs = np.einsum("ect,edt->ecd", E, E) / E.shape[-1]
        # robust epoch rejection: drop epochs with outlying total power (artifacts)
        lt = np.log(np.trace(covs, axis1=1, axis2=2) + 1e-12)
        med = np.median(lt)
        mad = np.median(np.abs(lt - med)) + 1e-6
        keep = lt <= med + 3.0 * 1.4826 * mad
        if keep.sum() < 1:
            keep[:] = True
        C = covs[keep].mean(axis=0)
        # shrinkage: average reference makes C rank-deficient
        C = 0.95 * C + 0.05 * np.trace(C) / n_ch * np.eye(n_ch)
        out.append(_logm_upper(C))
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
            f.append(_spectrum(x, fs))
            f.append(_covariances(x, fs))
            f.append(_variability(x, fs))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X, y = np.asarray(F_train, dtype=np.float64), np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    gkf = GroupKFold(n_splits=5)
    scores = np.zeros(len(CS))
    for tr, va in gkf.split(X, y, groups):
        for j, C in enumerate(CS):
            sc, clf = _fit(X[tr], y[tr], C)
            scores[j] += roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    best_C = CS[int(np.argmax(scores))]
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]
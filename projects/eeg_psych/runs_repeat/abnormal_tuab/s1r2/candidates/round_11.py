"""Log relative power spectrum in 1-Hz bins (1-45 Hz) per channel plus log absolute power in
5 classical bands per channel, from a median-averaged (artifact-robust) Welch PSD, plus
log-Euclidean tangent-space band covariance matrices (delta/theta/alpha/beta, robust epoch
averaging), plus temporal variability (quantile spreads) of 2-s epoch band power capturing
intermittent slowing / paroxysmal events, plus time-domain waveform-shape features (Hjorth
mobility/complexity, skewness, kurtosis of 2-s epochs) capturing sharp / epileptiform transients,
plus side-agnostic focality features (|left-right| homologous band-power asymmetry and extreme
per-channel deviation from the spatial median) capturing focal slowing / unilateral alpha loss,
plus spectral parameterization (aperiodic 1/f exponent/offset, peak alpha frequency, alpha peak
prominence, theta/delta bump on the 1/f-flattened spectrum) capturing posterior-rhythm slowing,
plus band-wise weighted phase-lag index (lagged, volume-conduction-robust phase synchrony) per
channel pair and per-channel node strength.
Model: blend of standardized L2 logistic regression and LightGBM (nonlinear thresholds /
interactions); C and blend weight chosen by inner grouped CV."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

BINS = [(k, k + 1) for k in range(1, 45)]
ABS_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
CS = [0.001, 0.003, 0.01, 0.03, 0.1]
WS = [0.0, 0.25, 0.5, 0.75, 1.0]   # weight on logistic regression in the blend
N_SHAPE = 6
N_PARAM = 7
HOMOLOGOUS = [("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("T3|T7", "T4|T8"), ("C3", "C4"),
              ("T5|P7", "T6|P8"), ("P3", "P4"), ("O1", "O2")]


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


def _param(x, fs):
    # spectral parameterization: a slowed posterior dominant rhythm (< 8 Hz) is a defining
    # criterion of an abnormal routine EEG. Peak position is an argmax / centroid of the
    # 1/f-flattened spectrum, which neither a linear model on 1-Hz bins nor sparse trees can form.
    n_ch, n_t = x.shape
    nper = int(min(4 * fs, n_t))
    f, p = welch(x, fs=fs, nperseg=nper, axis=-1, average="median")
    if len(f) < 8:
        return np.zeros(n_ch * N_PARAM + 3)
    lf = np.log10(np.maximum(f, 1e-6))
    lp = np.log10(p + 1e-12)
    m = (f >= 2) & (f <= 40) & ~((f >= 6) & (f <= 14))
    if m.sum() < 3:
        return np.zeros(n_ch * N_PARAM + 3)
    slope, icpt = np.polyfit(lf[m], lp[:, m].T, 1)                    # (ch,), (ch,)
    resid = lp - (slope[:, None] * lf[None] + icpt[:, None])
    # light smoothing of the flattened spectrum over ~0.75 Hz
    k = np.ones(3) / 3.0
    resid = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, resid)
    df = f[1] - f[0]
    a = (f >= 6) & (f <= 14)
    fa, ra = f[a], resid[:, a]
    imax = np.argmax(ra, axis=1)
    paf = fa[imax]
    peak_h = ra[np.arange(n_ch), imax]
    w = np.maximum(ra, 0.0)
    ws = w.sum(axis=1)
    cog = np.where(ws > 1e-9, (w * fa[None]).sum(axis=1) / np.maximum(ws, 1e-9), 10.0)
    area = ws * df
    t = (f >= 2) & (f < 6)
    slow_h = resid[:, t].max(axis=1)
    feats = np.stack([-slope, icpt, paf, peak_h, cog, area, slow_h], axis=1)
    summ = np.array([np.median(cog), np.max(cog), np.median(-slope)])
    return np.concatenate([feats.ravel(), summ])


def _resolve_pairs(channels):
    idx = {c.lower(): i for i, c in enumerate(channels)}
    pairs = []
    for left, right in HOMOLOGOUS:
        li = next((idx[n.lower()] for n in left.split("|") if n.lower() in idx), None)
        ri = next((idx[n.lower()] for n in right.split("|") if n.lower() in idx), None)
        if li is not None and ri is not None:
            pairs.append((li, ri))
    return pairs


def _focality(x, fs, pairs):
    # focal slowing / unilateral alpha attenuation may sit on either hemisphere or any
    # electrode: |L-R| and the extreme deviation from the spatial median are side-agnostic,
    # V-shaped functions of per-channel power that a linear model cannot form itself
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f < 45)].sum(axis=1) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in ABS_BANDS], axis=1)
    la = np.log(bp + 1e-12)
    lr = np.log(bp[:, :4] / tot[:, None] + 1e-12)
    L = np.concatenate([la, lr], axis=1)                              # (ch, 9)
    out = []
    if pairs:
        li = np.array([a for a, _ in pairs])
        ri = np.array([b for _, b in pairs])
        A = np.abs(L[li] - L[ri])                                     # (pairs, 9)
        out += [A.ravel(), A.max(axis=0)]
    dev = L - np.median(L, axis=0, keepdims=True)
    out += [dev.max(axis=0), dev.min(axis=0)]
    return np.concatenate(out)


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


def _shape(x, fs):
    # spikes / sharp waves are brief, steep and heavy-tailed: they barely move the spectrum but
    # raise kurtosis, skewness and Hjorth complexity of the epochs that contain them. All
    # features are amplitude-scale invariant (robust to amplifier gain on external sites).
    n_ch, n_t = x.shape
    ep = int(2 * fs)
    if n_t < ep:
        return np.zeros(n_ch * N_SHAPE)
    sos = butter(4, [1.0, min(45.0, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    n_ep = n_t // ep
    E = xf[:, : n_ep * ep].reshape(n_ch, n_ep, ep)
    E = E - E.mean(axis=-1, keepdims=True)
    d1 = np.diff(E, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = E.var(axis=-1) + 1e-12
    v1 = d1.var(axis=-1) + 1e-12
    v2 = d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    sd = np.sqrt(v0)[..., None]
    z = E / sd
    skew = (z ** 3).mean(axis=-1)
    kurt = np.log((z ** 4).mean(axis=-1) + 1e-6)
    lmob = np.log(mob)
    mq10, mq90 = np.percentile(lmob, [10, 90], axis=-1)
    feats = [
        np.median(lmob, axis=-1),
        np.median(np.log(comp), axis=-1),
        mq90 - mq10,
        np.median(skew, axis=-1),
        np.median(kurt, axis=-1),
        np.percentile(kurt, 90, axis=-1),
    ]
    return np.stack(feats, axis=1).ravel()


def _wpli(x, fs):
    # weighted phase-lag index: only consistently lagged (non-zero-phase) coupling contributes,
    # so it is insensitive to volume conduction / common reference that dominate the zero-lag
    # covariance, and to amplitude scale. Encephalopathy, lesions and epileptiform networks
    # alter lagged large-scale synchrony.
    n_ch, n_t = x.shape
    n = int(2 * fs)
    iu = np.triu_indices(n_ch, k=1)
    n_out = len(COV_BANDS) * (len(iu[0]) + n_ch)
    n_ep = n_t // n
    if n_ep < 4:
        return np.zeros(n_out)
    E = x[:, : n_ep * n].reshape(n_ch, n_ep, n)
    E = E - E.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(E * np.hanning(n)[None, None], axis=-1)           # (ch, ep, f)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # robust epoch rejection on total 1-45 Hz power (artifacts)
    pw = (np.abs(X[:, :, (f >= 1) & (f < 45)]) ** 2).sum(axis=(0, 2))
    lt = np.log(pw + 1e-12)
    med = np.median(lt)
    mad = np.median(np.abs(lt - med)) + 1e-6
    keep = lt <= med + 3.0 * 1.4826 * mad
    if keep.sum() < 4:
        keep[:] = True
    X = X[:, keep]
    out_pairs, out_nodes = [], []
    for lo, hi in COV_BANDS:
        Xb = X[:, :, (f >= lo) & (f < hi)]                            # (ch, ep, nf)
        im = np.einsum("iek,jek->ijek", Xb, np.conj(Xb)).imag          # (ch, ch, ep, nf)
        num = np.abs(im.sum(axis=2))
        den = np.abs(im).sum(axis=2) + 1e-12
        W = (num / den).mean(axis=-1)                                 # (ch, ch)
        np.fill_diagonal(W, 0.0)
        out_pairs.append(W[iu])
        out_nodes.append(W.sum(axis=1) / (n_ch - 1))
    return np.concatenate(out_pairs + out_nodes)


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
    pairs = _resolve_pairs(list(ctx["channels"]))
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
            f.append(_shape(x, fs))
            f.append(_focality(x, fs, pairs))
            f.append(_param(x, fs))
            f.append(_wpli(x, fs))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def _fit_gbm(X, y):
    # shallow, heavily subsampled trees: capture thresholded / interacting markers (e.g. focal
    # slowing in a subset of channels) that a linear model on standardized features misses
    gbm = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
        random_state=0, n_jobs=4, verbose=-1,
    )
    return gbm.fit(X, y)


def fit_predict(F_train, y_train, F_test, ctx):
    X, y = np.asarray(F_train, dtype=np.float64), np.asarray(y_train)
    Xt = np.asarray(F_test, dtype=np.float64)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))
    scores = np.zeros(len(CS))
    p_lr = np.zeros((len(CS), len(y)))
    p_gb = np.zeros(len(y))
    for tr, va in folds:
        for j, C in enumerate(CS):
            sc, clf = _fit(X[tr], y[tr], C)
            p_lr[j, va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
            scores[j] += roc_auc_score(y[va], p_lr[j, va])
        p_gb[va] = _fit_gbm(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    jb = int(np.argmax(scores))
    best_C = CS[jb]
    # blend weight chosen by mean inner-fold AUC of the probability average
    wscores = np.zeros(len(WS))
    for tr, va in folds:
        for k, w in enumerate(WS):
            wscores[k] += roc_auc_score(y[va], w * p_lr[jb, va] + (1 - w) * p_gb[va])
    w = WS[int(np.argmax(wscores))]
    sc, clf = _fit(X, y, best_C)
    pred = w * clf.predict_proba(sc.transform(Xt))[:, 1]
    if w < 1.0:
        pred = pred + (1 - w) * _fit_gbm(X, y).predict_proba(Xt)[:, 1]
    return pred
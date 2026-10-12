"""Log relative band power (delta/theta/alpha/beta/gamma + high-frequency 55-95 Hz EMG band) per channel
from the mean Welch PSD of each condition (relative to 1-45 Hz total power), plus temporal variability
(std over 2 s segments of log band power) capturing non-stationarity / movement / muscle-tension
fluctuations, plus electrode-contact instability (per-channel std over 2 s segments of the line-noise
peak relative to neighbouring 46-54 Hz power, and its scalp-median level: impedance fluctuations from
movement / sweat), plus ocular-event features (blink / saccade rate per minute, robust relative amplitude,
event duration and irregularity of inter-event intervals from frontal channels), plus global movement /
muscle burst features (rate per minute, fraction of time, robust magnitude of scalp-wide bursts in
0.1-1 Hz drift and 55-95 Hz EMG envelopes), plus slow eye-movement features (0.1-1 Hz power of bipolar
horizontal F7-F8 and vertical Fp-Cz derivations relative to their own 1-45 Hz power: overall, variability
over 4 s windows, episodic excess) capturing gaze wandering (EO) and drowsiness-related slow rolling eye
movements (EC), plus per-channel slow-drift (0.1-1 Hz) power relative to the channel's 1-45 Hz power and
its variability over 4 s windows (topography of eye / head / body movement drifts), plus spatial coupling
of the slow drift (per-channel mean inter-channel correlation and leading-eigenvalue variance fractions
of the 0.1-1 Hz correlation matrix: global movement vs local drift) and its time course (distribution of
the leading-eigenvalue fraction over 10 s windows: episodic scalp-wide movement), plus respiration-related
features of the leading slow-drift component (breathing-range spectral peak frequency, peak prominence,
peak power concentration and slow-band spectral entropy: breathing rate / regularity as arousal markers);
standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import spectrogram, butter, sosfiltfilt, find_peaks, peak_widths, welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
EMG_BAND = (55, 95)   # avoids 50 Hz and 100 Hz line-noise peaks; dominated by scalp muscle activity
EVENT_Z = 5.0         # robust-z threshold for ocular events
EVENT_GAP = 0.3       # s, minimum distance between events
DRIFT_BAND = (0.1, 1.0)  # slow movement / electrode / sweat drifts
BURST_WIN = 0.5       # s, window for log-RMS envelopes
BURST_Z = 3.0         # robust-z threshold (median over channels) for scalp-wide bursts
BURST_FRAC_Z = 2.0    # robust-z threshold for fraction-of-time feature
SEM_BAND = (0.1, 1.0)    # slow eye movements / gaze drift
SEM_WIN = 4.0         # s, window for slow-eye-movement envelopes
COUP_WIN = 10.0       # s, window for time-resolved slow-drift coupling
COUP_THR = 0.5        # leading-eigenvalue fraction marking a scalp-wide (movement) window
RESP_BAND = (0.12, 0.6)  # Hz, breathing range (~7-36 breaths/min)
RESP_SEG = 30.0       # s, Welch segment for the respiration spectrum
RESP_HALFBW = 0.05    # Hz, half-width around the peak for power concentration
LINE_HALFBW = 0.6     # Hz, half-width of the line-noise peak
LINE_NB = (1.0, 4.0)  # Hz, offsets of the neighbouring reference band on each side of the line


def _line_instability(f, S, fs):
    """Line-noise peak (50 or 60 Hz, whichever dominates) relative to neighbouring power, per 2 s
    segment and channel. Returns per-channel std over segments (impedance fluctuation) and the
    scalp-median of the mean log ratio (contact quality). Scale- and length-free."""
    nch = S.shape[0]
    cands = [lf for lf in (50.0, 60.0) if lf + LINE_NB[1] < 0.95 * fs / 2.0]
    if not cands:
        return np.zeros(nch + 1)
    p = S.mean(axis=(0, 2))
    best, best_r = cands[0], -np.inf
    for lf in cands:
        pk = p[np.abs(f - lf) <= LINE_HALFBW].sum()
        nb = p[(np.abs(f - lf) >= LINE_NB[0]) & (np.abs(f - lf) <= LINE_NB[1])].mean() + 1e-20
        r = pk / nb
        if r > best_r:
            best, best_r = lf, r
    pk = S[:, np.abs(f - best) <= LINE_HALFBW, :].mean(axis=1)
    nbm = (np.abs(f - best) >= LINE_NB[0]) & (np.abs(f - best) <= LINE_NB[1])
    nb = S[:, nbm, :].mean(axis=1)
    L = np.log(pk + 1e-20) - np.log(nb + 1e-20)   # ch x T
    var = L.std(axis=1) if L.shape[1] > 1 else np.zeros(nch)
    lvl = np.median(L.mean(axis=1))
    return np.concatenate([var, [lvl]])


def _features(x, fs):
    nper = int(2 * fs)
    f, _, S = spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                          detrend="constant", scaling="density", mode="psd", axis=-1)
    bands = list(BANDS)
    hi = min(EMG_BAND[1], 0.95 * fs / 2.0)
    bands.append((EMG_BAND[0], hi) if hi > EMG_BAND[0] + 5 else (45, 0.95 * fs / 2.0))
    # S: (channels, freqs, segments); mean over segments == Welch mean PSD
    seg_bp = np.stack([S[:, (f >= a) & (f < b), :].sum(axis=1) for a, b in bands], axis=1)  # ch x band x T
    p = S.mean(axis=2)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(seg_bp.mean(axis=2) / tot + 1e-12)
    if seg_bp.shape[2] > 1:
        var = np.log(seg_bp + 1e-12).std(axis=2)
    else:
        var = np.zeros_like(rel)
    return np.concatenate([rel.ravel(), var.ravel(), _line_instability(f, S, fs)])


def _ocular(x, fs, chans):
    """Vertical (Fp1/Fp2 mean, amplitude) and horizontal (F7-F8, velocity) eye-event rate,
    robust relative amplitude, log median event duration (half-height width, s) and irregularity
    of inter-event intervals (std of log intervals). Scale-free (MAD-normalized) and length-free."""
    idx = {c.lower(): i for i, c in enumerate(chans)}
    get = lambda names: [idx[n.lower()] for n in names if n.lower() in idx]
    vi = get(["Fp1", "Fp2"]) or [0]
    v = x[vi].mean(axis=0)
    hl, hr = get(["F7"]), get(["F8"])
    h = x[hl[0]] - x[hr[0]] if (hl and hr) else x[vi[0]]
    sos = butter(4, [0.5, 6.0], btype="band", fs=fs, output="sos")
    dur_min = max(x.shape[1] / fs / 60.0, 1e-3)
    out = []
    for sig, deriv, w0 in ((v, False, 0.25), (h, True, 0.05)):
        s = sosfiltfilt(sos, sig)
        if deriv:
            s = np.gradient(s) * fs
        s = s - np.median(s)
        sd = 1.4826 * np.median(np.abs(s)) + 1e-9
        z = np.abs(s) / sd
        pk, props = find_peaks(z, height=EVENT_Z, distance=max(1, int(EVENT_GAP * fs)))
        rate = len(pk) / dur_min
        amp = np.log(np.median(props["peak_heights"])) if len(pk) else np.log(EVENT_Z)
        if len(pk):
            w = peak_widths(z, pk, rel_height=0.5)[0] / fs
            width = np.log(np.median(w) + 1e-3)
        else:
            width = np.log(w0)
        if len(pk) >= 4:
            iei = np.diff(pk) / fs
            irreg = np.log(iei + 1e-3).std()
        else:
            irreg = 0.0
        out += [np.log1p(rate), amp, width, irreg]
    return np.asarray(out)


def _slow_eye(x, fs, chans):
    """Slow (0.1-1 Hz) eye-movement activity of bipolar horizontal (F7-F8) and vertical (Fp-Cz)
    derivations, as log RMS relative to the same derivation's 1-45 Hz RMS (scale/reference-free):
    overall ratio, std over 4 s windows, and 90th percentile minus median of the windowed ratio."""
    idx = {c.lower(): i for i, c in enumerate(chans)}
    get = lambda names: [idx[n.lower()] for n in names if n.lower() in idx]
    fp = get(["Fp1", "Fp2"]) or [0]
    cz = get(["Cz"]) or get(["C3", "C4"]) or [x.shape[0] - 1]
    hl, hr = get(["F7"]), get(["F8"])
    h = x[hl[0]] - x[hr[0]] if (hl and hr) else x[fp[0]] - x[fp[-1]]
    v = x[fp].mean(axis=0) - x[cz].mean(axis=0)
    sos_s = butter(2, list(SEM_BAND), btype="band", fs=fs, output="sos")
    sos_b = butter(4, [1.0, min(45.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    win = max(1, int(SEM_WIN * fs))
    out = []
    for sig in (h, v):
        sig = sig - np.median(sig)
        s = sosfiltfilt(sos_s, sig)
        b = sosfiltfilt(sos_b, sig)
        ratio = 0.5 * np.log((np.mean(s ** 2) + 1e-12) / (np.mean(b ** 2) + 1e-12))
        n = len(sig) // win
        if n >= 3:
            Ls = 0.5 * np.log(np.mean(s[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
            Lb = 0.5 * np.log(np.mean(b[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
            r = Ls - Lb
            out += [ratio, r.std(), np.percentile(r, 90) - np.median(r)]
        else:
            out += [ratio, 0.0, 0.0]
    return np.asarray(out)


def _lead_frac(z):
    """Fraction of variance in the leading eigenvalue of the channel correlation matrix of z."""
    z = z - z.mean(axis=1, keepdims=True)
    z = z / (z.std(axis=1, keepdims=True) + 1e-12)
    R = (z @ z.T) / max(z.shape[1], 1)
    ev = np.linalg.eigvalsh(R)
    return ev[-1] / (ev.sum() + 1e-12)


def _respiration(pc, fs_d):
    """Respiration-related spectrum of the leading slow-drift component: peak frequency in the
    breathing range, log peak-to-median prominence, fraction of 0.1-1 Hz power within +-0.05 Hz
    of the peak (regularity), and normalized spectral entropy of the 0.1-1 Hz spectrum.
    Frequency / ratio based: amplitude-scale-free and length-free (fixed Welch segment)."""
    default = np.array([0.25, 0.0, 0.0, 1.0])
    nper = min(int(RESP_SEG * fs_d), len(pc) // 2)
    if nper < int(10 * fs_d):
        return default
    f, p = welch(pc, fs=fs_d, window="hann", nperseg=nper, noverlap=nper // 2, detrend="constant")
    sb = (f >= SEM_BAND[0]) & (f <= SEM_BAND[1])
    rb = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    if sb.sum() < 4 or rb.sum() < 2:
        return default
    ps = p[sb] + 1e-20
    pr = p[rb]
    fr = f[rb]
    k = int(np.argmax(pr))
    fpk = fr[k]
    prom = np.log(pr[k] / np.median(ps) + 1e-12)
    conc = p[(f >= fpk - RESP_HALFBW) & (f <= fpk + RESP_HALFBW)].sum() / ps.sum()
    q = ps / ps.sum()
    ent = -(q * np.log(q)).sum() / np.log(len(q))
    return np.array([fpk, prom, conc, ent])


def _slow_coupling(s, fs):
    """Spatial coupling of slow (0.1-1 Hz) drift: per-channel mean Pearson correlation with all
    other channels, fractions of variance in the two leading eigenvalues of the inter-channel
    correlation matrix, and the distribution over 10 s windows (50% overlap) of the leading-
    eigenvalue fraction (median, std, 90th percentile, fraction of windows above 0.5) capturing
    episodic scalp-wide movement, plus respiration features of the leading component.
    Correlation-based, hence amplitude-scale-free and length-free."""
    step = max(1, int(fs // 10))          # slow band < 1 Hz -> 10 Hz sampling is ample
    z = s[:, ::step]
    z = z - z.mean(axis=1, keepdims=True)
    z = z / (z.std(axis=1, keepdims=True) + 1e-12)
    n = z.shape[0]
    R = (z @ z.T) / max(z.shape[1], 1)
    mean_r = (R.sum(axis=1) - np.diag(R)) / max(n - 1, 1)
    evals, evecs = np.linalg.eigh(R)
    ev = evals[::-1]
    frac = ev[:2] / (ev.sum() + 1e-12)
    if frac.size < 2:
        frac = np.pad(frac, (0, 2 - frac.size))
    fs_d = fs / step
    win = max(2, int(COUP_WIN * fs_d))
    hop = max(1, win // 2)
    starts = range(0, z.shape[1] - win + 1, hop)
    lf = np.array([_lead_frac(z[:, a:a + win]) for a in starts])
    if lf.size >= 3:
        tv = np.array([np.median(lf), lf.std(), np.percentile(lf, 90), np.mean(lf > COUP_THR)])
    else:
        tv = np.array([frac[0], 0.0, frac[0], float(frac[0] > COUP_THR)])
    pc = evecs[:, -1] @ z
    resp = _respiration(pc, fs_d)
    return np.concatenate([mean_r, frac, tv, resp])


def _slow_channels(x, fs):
    """Per-channel slow-drift (0.1-1 Hz) log RMS relative to the channel's own 1-45 Hz RMS
    (scale-free), std of that windowed ratio over 4 s windows (length-free), and spatial
    coupling of the slow drift across channels."""
    sos_s = butter(2, list(SEM_BAND), btype="band", fs=fs, output="sos")
    sos_b = butter(4, [1.0, min(45.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    xc = x - np.median(x, axis=1, keepdims=True)
    s = sosfiltfilt(sos_s, xc, axis=-1)
    b = sosfiltfilt(sos_b, xc, axis=-1)
    ratio = 0.5 * np.log((np.mean(s ** 2, axis=1) + 1e-12) / (np.mean(b ** 2, axis=1) + 1e-12))
    win = max(1, int(SEM_WIN * fs))
    n = x.shape[1] // win
    if n >= 3:
        Ls = 0.5 * np.log(np.mean(s[:, :n * win].reshape(x.shape[0], n, win) ** 2, axis=2) + 1e-12)
        Lb = 0.5 * np.log(np.mean(b[:, :n * win].reshape(x.shape[0], n, win) ** 2, axis=2) + 1e-12)
        var = (Ls - Lb).std(axis=1)
    else:
        var = np.zeros_like(ratio)
    return np.concatenate([ratio, var, _slow_coupling(s, fs)])


def _bursts(x, fs):
    """Scalp-wide burst features for slow drift (movement) and EMG (muscle) envelopes:
    per-channel log-RMS in 0.5 s windows, robust-z per channel, median across channels
    (so only events hitting many channels count). Returns log rate/min, fraction of time
    above threshold and log median peak z per band. Scale- and length-free."""
    win = max(1, int(BURST_WIN * fs))
    nwin = x.shape[1] // win
    dur_min = max(x.shape[1] / fs / 60.0, 1e-3)
    hi = min(EMG_BAND[1], 0.95 * fs / 2.0)
    emg = (EMG_BAND[0], hi) if hi > EMG_BAND[0] + 5 else (45, 0.95 * fs / 2.0)
    out = []
    for (lo, hb), order in ((DRIFT_BAND, 2), (emg, 4)):
        if nwin < 4:
            out += [0.0, 0.0, np.log(BURST_Z)]
            continue
        sos = butter(order, [lo, hb], btype="band", fs=fs, output="sos")
        s = sosfiltfilt(sos, x, axis=-1)[:, :nwin * win]
        L = 0.5 * np.log(np.mean(s.reshape(s.shape[0], nwin, win) ** 2, axis=2) + 1e-12)  # ch x nwin
        med = np.median(L, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(L - med), axis=1, keepdims=True) + 1e-9
        g = np.median((L - med) / mad, axis=0)
        pk, props = find_peaks(g, height=BURST_Z, distance=2)
        rate = len(pk) / dur_min
        frac = float(np.mean(g > BURST_FRAC_Z))
        amp = np.log(np.median(props["peak_heights"])) if len(pk) else np.log(BURST_Z)
        out += [np.log1p(rate), np.sqrt(frac), amp]
    return np.asarray(out)


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    chans = list(ctx["channels"])
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_features(x, fs))
            f.append(_ocular(x, fs, chans))
            f.append(_bursts(x, fs))
            f.append(_slow_eye(x, fs, chans))
            f.append(_slow_channels(x, fs))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]
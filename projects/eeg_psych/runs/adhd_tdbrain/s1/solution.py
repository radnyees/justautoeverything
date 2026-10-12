"""Seed log relative band power per channel (Welch, EC+EO) + global temporal variability across 2-s
segments of (a) log absolute band power, (b) log relative band power (spectral-composition
instability, e.g. vigilance-related alpha->theta shifts, insensitive to broadband amplitude bursts),
and (c) all pairwise log band ratios (the compositional / Aitchison variation matrix: theta/beta,
theta/alpha, alpha/beta arousal, delta/alpha drowsiness, ratios vs. EMG-dominated 30-45 Hz for
motor restlessness). The variability block uses a finer band partition (low/high alpha, low/high
beta) to resolve vigilance-stage shifts (alpha slowing, beta1/beta2 drowsiness changes).
Standardized, block-weighted (equal total variance per feature block) L2 logistic regression."""
from itertools import combinations

import numpy as np
from scipy.signal import welch, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
# finer partition for the temporal-variability block
VBANDS = [(1, 4), (4, 8), (8, 10.5), (10.5, 13), (13, 20), (20, 30), (30, 45)]
RATIOS = list(combinations(range(len(VBANDS)), 2))   # all pairwise log ratios (variation matrix)
N_VAR = 2 * len(VBANDS) + len(RATIOS)   # variability features per condition


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _variability(x, fs):
    # SD over 2-s segments of log band power (absolute and relative) and of all pairwise log band
    # ratios, per channel, averaged over channels -> (N_VAR,)
    n = int(2 * fs)
    f, _, s = spectrogram(x, fs=fs, nperseg=n, noverlap=n // 2, axis=-1)   # (ch, freq, seg)
    ltot = np.log(s[:, (f >= 1) & (f <= 45), :].sum(axis=1) + 1e-12)      # (ch, seg)
    out_abs, out_rel, lbp = [], [], []
    for a, b in VBANDS:
        bp = np.log(s[:, (f >= a) & (f < b), :].sum(axis=1) + 1e-12)       # (ch, seg)
        lbp.append(bp)
        out_abs.append(np.mean(np.std(bp, axis=1)))
        out_rel.append(np.mean(np.std(bp - ltot, axis=1)))
    out_ratio = [np.mean(np.std(lbp[i] - lbp[j], axis=1)) for i, j in RATIOS]
    return np.asarray(out_abs + out_rel + out_ratio)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_bandpower(x, ctx["fs"]).ravel())
            f.append(_variability(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _block_weights(n_feat, ctx):
    # layout per condition: [n_ch * n_bands band powers, N_VAR variability]
    n_bp = len(ctx["channels"]) * len(BANDS)
    per_cond = n_bp + N_VAR
    is_var = np.zeros(n_feat, dtype=bool)
    for k in range(n_feat // per_cond):
        is_var[k * per_cond + n_bp:(k + 1) * per_cond] = True
    w = np.ones(n_feat)
    if is_var.any() and (~is_var).any():
        w[is_var] = np.sqrt((~is_var).sum() / is_var.sum())   # equal total variance per block
    return w


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    w = _block_weights(F_train.shape[1], ctx)
    Xtr = sc.transform(F_train) * w
    Xte = sc.transform(F_test) * w
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
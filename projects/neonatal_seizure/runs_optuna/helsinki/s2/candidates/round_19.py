PARAMS = {'montage': 'bipolar', 'win': 16, 'll': False, 'rms': False, 'bp': True, 'entropy': True, 'rhythm': False, 'neonate_norm': True, 'model': 'hgb', 'iters': 200, 'smooth': 21}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BIPOLAR = [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("Fp1", "F3"), ("F3", "C3"),
           ("C3", "P3"), ("P3", "O1"), ("Fp2", "T4"), ("T4", "O2"), ("Fp1", "T3"), ("T3", "O1"),
           ("T4", "C4"), ("C4", "Cz"), ("Cz", "C3"), ("C3", "T3"), ("Fz", "Cz"), ("Cz", "Pz")]
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


def _features(x, fs, nsec, chans):
    if PARAMS["montage"] == "bipolar":
        alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        chans = [alias.get(c, c) for c in chans]
        x = np.stack([x[chans.index(a)] - x[chans.index(b)] for a, b in BIPOLAR
                      if a in chans and b in chans])
    win = PARAMS["win"] * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    rows = []
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win]
        seg = seg - seg.mean(axis=1, keepdims=True)
        f = []
        if PARAMS["ll"]:
            f.append(np.abs(np.diff(seg, axis=1)).mean(axis=1))
        if PARAMS["rms"]:
            f.append(np.sqrt((seg ** 2).mean(axis=1)))
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 + 1e-9
        if PARAMS["bp"]:
            f.extend([P[:, m].sum(axis=1) for m in masks])
        if PARAMS["entropy"]:
            q = P[:, (freqs > 0.5) & (freqs < 30)]
            q = q / q.sum(axis=1, keepdims=True)
            f.append(np.exp(-(q * np.log(q)).sum(axis=1)))
        if PARAMS["rhythm"]:                           # spectral peakiness 0.5-8 Hz
            q = P[:, (freqs > 0.5) & (freqs < 8)]
            f.append(q.max(axis=1) / q.mean(axis=1))
        per_ch = np.log(np.column_stack(f) + 1e-6)
        rows.append(np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0),
                                    np.sort(per_ch, axis=0)[-3:].mean(axis=0)]))
    F = np.asarray(rows, dtype=np.float32)
    if PARAMS["neonate_norm"]:
        F = (F - np.median(F, axis=0)) / (F.std(axis=0) + 1e-6)
    return F


def featurize(items, ctx):
    fs, chans = ctx["fs"], list(ctx["channels"])
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it), chans)
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    X, y = np.vstack(F_train), np.concatenate(y_train)
    m = PARAMS["model"]
    if m == "rf":
        idx = np.random.default_rng(0).choice(len(y), min(len(y), 120000), replace=False)
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0)
        clf.fit(X[idx], y[idx])
    elif m == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)
    else:
        clf = HistGradientBoostingClassifier(max_iter=PARAMS["iters"], learning_rate=0.1, random_state=0).fit(X, y)
    k = np.ones(PARAMS["smooth"]) / PARAMS["smooth"]
    return [np.convolve(clf.predict_proba(F)[:, 1], k, mode="same") for F in F_test]

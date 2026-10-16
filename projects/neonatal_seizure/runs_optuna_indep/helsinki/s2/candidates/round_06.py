PARAMS = {'win': 4, 'smooth': 1, 'll': True, 'rms': True, 'bp': False, 'agg': 'max', 'model': 'hgb', 'C': 0.7188834654786145, 'rf_trees': 1000, 'rf_leaf': 3, 'hgb_lr': 0.03006733013951688, 'hgb_depth': 3, 'hgb_iter': 100}
"""Seed-derived (independent) Optuna template: the seed (per-second features on a sliding window: line
length, RMS, 4 band powers per channel, aggregated over channels -> histogram gradient boosting ->
moving-average smoothing) with its constants tunable, its feature groups switchable, plus the generic
model list (docs/preregistration_phase3.md). PARAMS is prepended by core/search.py."""
import numpy as np
from core.generic_models import make_classifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


def _features(x, fs, nsec):
    win = PARAMS["win"] * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    rows = []
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win]
        seg = seg - seg.mean(axis=1, keepdims=True)
        cols = []
        if PARAMS["ll"]:
            cols.append(np.abs(np.diff(seg, axis=1)).mean(axis=1))
        if PARAMS["rms"]:
            cols.append(np.sqrt((seg ** 2).mean(axis=1)))
        if PARAMS["bp"] or not cols:                     # at least one feature group
            P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
            cols.extend(P[:, m].sum(axis=1) for m in masks)
        per_ch = np.log(np.column_stack(cols) + 1e-6)
        agg = {"max+median": [per_ch.max(axis=0), np.median(per_ch, axis=0)],
               "max": [per_ch.max(axis=0)], "median": [np.median(per_ch, axis=0)]}[PARAMS["agg"]]
        rows.append(np.concatenate(agg))
    return np.asarray(rows, dtype=np.float32)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    clf = make_classifier(PARAMS).fit(np.vstack(F_train), np.concatenate(y_train))
    k = np.ones(PARAMS["smooth"]) / PARAMS["smooth"]
    return [np.convolve(clf.predict_proba(F)[:, 1], k, mode="same") for F in F_test]

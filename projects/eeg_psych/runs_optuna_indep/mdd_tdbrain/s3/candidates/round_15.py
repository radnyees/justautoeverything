PARAMS = {'welch_sec': 2, 'fmax': 45, 'conds': 'first', 'model': 'hgb', 'C': 0.012727563644322669, 'rf_trees': 200, 'rf_leaf': 10, 'hgb_lr': 0.07842385329646127, 'hgb_depth': 3, 'hgb_iter': 300}
"""Seed-derived (independent) Optuna template: the seed (log relative band power per channel from Welch
PSD of each condition, standardization + L2 logistic regression) with its constants tunable, conditions
chosen among those the task exposes, plus the generic model list (docs/preregistration_phase3.md)."""
import numpy as np
from scipy.signal import welch
from core.generic_models import make_classifier

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(PARAMS["welch_sec"] * fs), axis=-1)
    bands = [b for b in BANDS if b[1] <= PARAMS["fmax"]]
    tot = p[:, (f >= 1) & (f <= PARAMS["fmax"])].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in bands], axis=1) / tot)


def featurize(items, ctx):
    conds = ctx["conditions"] if PARAMS["conds"] == "all" else ctx["conditions"][:1]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    feats = []
    for i in range(len(items)):
        f = []
        for c in conds:
            x = blocks[c][i]
            if x is None:
                x = blocks[conds[0]][i]
            f.append(_bandpower(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    clf = make_classifier(PARAMS).fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]

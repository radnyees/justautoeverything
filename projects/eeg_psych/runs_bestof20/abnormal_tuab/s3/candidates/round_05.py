"""Log absolute + relative band power (9 finer bands) per channel and channel-averaged, from a
median-averaged Welch PSD (robust to transient artifacts); LightGBM gradient boosting to capture
non-linear / focal slowing patterns typical of abnormal clinical EEG."""
import numpy as np
from scipy.signal import welch
from lightgbm import LGBMClassifier

BANDS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]


def _spec_feats(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    df = f[1] - f[0]
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) * df for a, b in BANDS], axis=1) + 1e-12
    tot = p[:, (f >= 1) & (f < 45)].sum(axis=1) * df + 1e-12
    log_abs = np.log(bp)                                   # ch x bands
    log_rel = np.log(bp / tot[:, None])                    # ch x bands
    log_tot = np.log(tot)                                  # ch
    g_abs = log_abs.mean(axis=0)                           # global (channel-averaged) bands
    g_rel = log_rel.mean(axis=0)
    return np.concatenate([log_abs.ravel(), log_rel.ravel(), log_tot, g_abs, g_rel,
                           [log_tot.mean()]])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spec_feats(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    clf = LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=0, n_jobs=4, verbose=-1, deterministic=True, force_col_wise=True,
    )
    clf.fit(np.asarray(F_train), np.asarray(y_train))
    return clf.predict_proba(np.asarray(F_test))[:, 1]
"""Log relative power in narrow 2-Hz bins (1-45 Hz) per channel from Welch PSD of each available
condition, standardized, L2 logistic regression with C chosen by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 2)            # 1,3,...,45 -> 22 bins of 2 Hz
CS = [0.001, 0.003, 0.01, 0.03, 0.1]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1)
    return np.log(bp / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    y_train = np.asarray(y_train)
    groups = np.asarray(ctx.get("groups_train", np.arange(len(y_train))))
    n_splits = min(5, len(np.unique(groups)))
    best_C, best_s = 0.1, -np.inf
    if n_splits >= 2:
        cv = list(GroupKFold(n_splits=n_splits).split(F_train, y_train, groups))
        for C in CS:
            oof = np.zeros(len(y_train))
            for tr, va in cv:
                oof[va] = _model(C).fit(F_train[tr], y_train[tr]).predict_proba(F_train[va])[:, 1]
            s = roc_auc_score(y_train, oof)
            if s > best_s:
                best_s, best_C = s, C
    clf = _model(best_C).fit(F_train, y_train)
    return clf.predict_proba(np.asarray(F_test))[:, 1]
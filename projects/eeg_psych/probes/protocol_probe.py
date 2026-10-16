"""TDBRAIN protocol probe (docs/preregistration_phase3.md, section 2). A harness solution whose features
are recording properties, not brain signals. PARAMS = {"set": "P1" | "P2"} is prepended by core/probe.py.
P1 (technical only), per condition (EC, EO): recording length; condition-present flag; line-noise power
49-51 Hz and 59-61 Hz relative to 1-45 Hz (median over channels); DC offset (median |channel mean|);
fraction of flat channels (SD < 0.5 uV); fraction of clipped samples (samples equal to the channel max or
min, mean over channels). P2 adds: log median channel SD; log power 55-100 Hz excluding 58-62 Hz (median
over channels); log total power 1-45 Hz (median over channels). Model: standardization + L2 logistic
regression, C = 1.0, no tuning."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _props(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    band = lambda a, b: p[:, (f >= a) & (f <= b)].sum(axis=1)               # noqa: E731
    tot = band(1, 45) + 1e-12
    sd = x.std(axis=1)
    clip = np.mean([np.mean((c == c.max()) | (c == c.min())) for c in x])
    p1 = [x.shape[1] / fs, 1.0, np.median(band(49, 51) / tot), np.median(band(59, 61) / tot),
          np.median(np.abs(x.mean(axis=1))), np.mean(sd < 0.5), clip]
    if PARAMS["set"] == "P1":
        return p1
    emg = band(55, 100) - band(58, 62)
    return p1 + [np.log(np.median(sd) + 1e-9), np.log(np.median(emg) + 1e-12), np.log(np.median(tot))]


def featurize(items, ctx):
    fs, rows = ctx["fs"], []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    n = 7 if PARAMS["set"] == "P1" else 10
    for i in range(len(items)):
        r = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            r += _props(x, fs) if x is not None else [0.0] * n          # flag (index 1) = 0 if absent
        rows.append(r)
    return np.nan_to_num(np.asarray(rows, dtype=np.float64))


def fit_predict(F_train, y_train, F_test, ctx):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]

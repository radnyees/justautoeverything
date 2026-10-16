
# ---- appended by core/probe_harmonized.py (EXPLORATORY, not pre-registered) ------------------------
# Wraps the solution's featurize so that every recording is harmonized before the solution sees it:
# identical length (first HARMONIZE_SEC seconds), mains noise removed (notch 50 and 60 Hz, and their
# harmonics below Nyquist), per-channel DC offset removed; optionally (HARMONIZE_SCALE) each recording
# scaled to unit median channel SD. The solution code itself is unchanged.
import numpy as _np
from scipy.signal import filtfilt as _filtfilt, iirnotch as _iirnotch

HARMONIZE_SEC = 100
try:
    HARMONIZE_SCALE                                   # noqa: B018  (set by core/probe_harmonized.py)
except NameError:
    HARMONIZE_SCALE = False                           # True: also scale each recording to unit median SD


def _harmonize(x, fs):
    x = _np.asarray(x, dtype=_np.float64)[:, : int(HARMONIZE_SEC * fs)]
    for f0 in (50.0, 60.0, 100.0, 120.0):
        if f0 < fs / 2 - 1:
            b, a = _iirnotch(f0, 30.0, fs)
            x = _filtfilt(b, a, x, axis=-1)
    x = x - x.mean(axis=1, keepdims=True)
    if HARMONIZE_SCALE:                               # removes overall amplitude (gain/impedance or biology)
        x = x / (_np.median(x.std(axis=1)) + 1e-9)
    return x.astype(_np.float32)


_orig_featurize = featurize


def featurize(items, ctx):
    ctx = dict(ctx)
    _load, _fs = ctx["load_eeg"], ctx["fs"]

    def load(*a, **k):
        return [None if x is None else _harmonize(x, _fs) for x in _load(*a, **k)]
    ctx["load_eeg"] = load
    return _orig_featurize(items, ctx)

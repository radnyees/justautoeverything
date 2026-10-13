"""
projects/eeg_psych/data/aseeg_channel_order.py -- infer the (undocumented) channel order of ASEEG's
preprocessed .mat files (19 x 38400, 128 Hz, 1-45 Hz) by matching its group-average inter-channel
correlation matrix to that of RepOD (the "Warsaw" montage ASEEG says it follows; labelled EDF
channels), after the same 1-45 Hz filter at 128 Hz. Scores standard candidate orders and a
swap-search (quadratic assignment) optimum; writes the result to /data/eeg_psych/aseeg_channel_order.json.
    python projects/eeg_psych/data/aseeg_channel_order.py /data/eeg_psych
"""
import glob
import itertools
import json
import sys

import h5py
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

base = sys.argv[1]
STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
         "P4", "P8", "O1", "O2"]
sos = butter(4, [1, 45], btype="band", fs=128, output="sos")


def corr(x):
    x = sosfiltfilt(sos, x, axis=1)
    return np.corrcoef(x)


# ASEEG group-average correlation (EC)
A = []
for f in sorted(glob.glob(f"{base}/raw/aseeg/EC/*/*.mat")):
    with h5py.File(f) as h:
        x = np.asarray(h["preprocessed_data"]).T            # 19 x samples
    A.append(corr(x))
A = np.mean(A, axis=0)

# RepOD group-average correlation in STD19 order (labelled), resampled to 128 Hz
R = []
for f in sorted(glob.glob(f"{base}/repod/*.npz")):
    x = np.load(f)["EC"]
    x = mne.filter.resample(x.astype(np.float64), up=128, down=250, axis=-1)
    R.append(corr(x))
R = np.mean(R, axis=0)
info = json.load(open(f"{base}/repod/info.json"))
idx = [info["channels"].index(c) for c in STD19]
R = R[np.ix_(idx, idx)]


def score(perm):                       # perm[i] = STD19 index of ASEEG column i
    P = np.asarray(perm)
    M = np.empty_like(A)
    M[np.ix_(P, P)] = A                # place ASEEG channels at their hypothesised labels
    iu = np.triu_indices(19, 1)
    return float(np.corrcoef(M[iu], R[iu])[0, 1])


def as_perm(order):
    m = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return [STD19.index(m.get(c, c)) for c in order]


cands = {
    "standard_10-20_rowwise": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                               "T5", "P3", "Pz", "P4", "T6", "O1", "O2"],
    "edf_bipolar_pairs": ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3",
                          "T4", "T5", "T6", "Fz", "Cz", "Pz"],
    "repod_file_order": [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(c, c) for c in info["channels"]],
    "warsaw_like": ["Fp2", "F8", "T4", "T6", "O2", "Fp1", "F7", "T3", "T5", "O1", "F4", "C4", "P4", "F3",
                    "C3", "P3", "Fz", "Cz", "Pz"],
}
res = {k: score(as_perm(v)) for k, v in cands.items()}

# unconstrained swap search from the best candidate
best_name = max(res, key=res.get)
perm, best = as_perm(cands[best_name]), res[best_name]
improved = True
while improved:
    improved = False
    for i, j in itertools.combinations(range(19), 2):
        p = list(perm)
        p[i], p[j] = p[j], p[i]
        s = score(p)
        if s > best + 1e-6:
            perm, best, improved = p, s, True
search_order = [STD19[k] for k in perm]
print("candidate scores", {k: round(v, 4) for k, v in res.items()})
print("best candidate", best_name, "| swap-search optimum", round(best, 4), search_order)
print("search == best candidate:", search_order == [STD19[k] for k in as_perm(cands[best_name])])
json.dump({"order": search_order, "match_r": best, "candidates": res, "best_candidate": best_name},
          open(f"{base}/aseeg_channel_order.json", "w"), indent=1)

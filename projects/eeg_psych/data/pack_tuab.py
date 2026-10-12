"""
projects/eeg_psych/data/pack_tuab.py -- pack the per-recording TUAB npz files into one float32 array
/data/eeg_psych/tuab/REST.npy (recordings x 19 x 30000, zero-padded) + REST_index.json
{subject: [row, n_samples]} so a candidate can memory-map the corpus instead of opening ~3,000 files.
    python projects/eeg_psych/data/pack_tuab.py /data/eeg_psych/tuab
"""
import json
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
p = pd.read_csv(f"{d}/participants.csv")
L = 30000
arr = np.lib.format.open_memmap(f"{d}/REST.npy", mode="w+", dtype=np.float32, shape=(len(p), 19, L))
index = {}
for i, s in enumerate(p["subject"]):
    x = np.load(f"{d}/{s}.npz")["REST"]
    n = min(x.shape[1], L)
    arr[i, :, :n] = x[:, :n]
    index[s] = [i, int(n)]
arr.flush()
json.dump(index, open(f"{d}/REST_index.json", "w"))
print("packed", arr.shape, flush=True)

"""
projects/eeg_psych/data/build_tuab.py -- TUH Abnormal EEG Corpus v3.0.1 (Obeid & Picone 2016; NEDC)
-> /data/eeg_psych/tuab/<rec>.npz {REST: float32 microvolts, 19 x 30000} + participants.csv.

Per recording (routine clinical EEG, average-reference 'tcp_ar' montage directory): the 19
standard 10-20 electrodes (modern names), the 5 minutes after the first 60 s (Gemein et al.
2020-style cropping), resampled to 100 Hz, clipped to +/-800 microvolts. The official TUAB
train/eval partition is kept: 'train' recordings form the task; 'eval' is an external test set.
    python projects/eeg_psych/data/build_tuab.py /data/eeg_psych
"""
import glob
import json
import os
import re
import sys

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
         "P4", "P8", "O1", "O2"]
OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
FS, START, DUR = 100, 60, 300


def convert(f, out):
    parts = f.split(os.sep)
    split = "eval" if "eval" in parts else "train"
    label = "abnormal" if "abnormal" in parts else "normal"
    rec = os.path.basename(f)[:-4]
    try:
        raw = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
        names = {re.sub(r"^EEG\s+", "", c).split("-")[0].upper(): c for c in raw.ch_names}
        picks = [names[OLD.get(c, c).upper()] for c in STD19]
        sf = raw.info["sfreq"]
        if raw.times[-1] < START + 60:              # need at least 1 min after the crop start
            return None
        raw.crop(tmin=START, tmax=min(START + DUR, raw.times[-1]) - 1 / sf)
        raw.pick(picks).load_data(verbose="ERROR")
        raw.resample(FS, verbose="ERROR")
        x = np.clip(raw.get_data() * 1e6, -800, 800).astype(np.float32)
    except Exception as e:
        print("skip", rec, type(e).__name__, str(e)[:80], flush=True)
        return None
    np.savez(f"{out}/{rec}.npz", REST=x)
    return {"subject": rec, "patient": rec.split("_")[0], "split": split, "status": label,
            "seconds": x.shape[1] / FS}


def main(base):
    out = f"{base}/tuab"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(f"{base}/raw/tuab/**/*.edf", recursive=True))
    print("edf files", len(files), flush=True)
    rows = Parallel(n_jobs=16)(delayed(convert)(f, out) for f in files)
    d = pd.DataFrame([r for r in rows if r])
    d.to_csv(f"{out}/participants.csv", index=False)
    json.dump({"channels": STD19, "fs": FS}, open(f"{out}/info.json", "w"))
    print(d.groupby(["split", "status"]).size().to_dict(), "patients", d["patient"].nunique(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""
projects/eeg_psych/data/build_open.py -- convert the open resting-state EEG datasets (downloaded by
fetch_open.py into /data/eeg_psych/raw/) to the common format:
    /data/eeg_psych/<dataset>/<subject>.npz  {EC, EO}: float32 microvolts, channels x samples
    /data/eeg_psych/<dataset>/participants.csv (subject, status, age, sex)
    /data/eeg_psych/<dataset>/info.json        {channels (modern 10-20 names), fs}
Channel names are mapped to modern 10-20 (T3->T7, T4->T8, T5->P7, T6->P8).
    python projects/eeg_psych/data/build_open.py /data/eeg_psych [mumtaz repod msu aseeg]
"""
import glob
import json
import os
import re
import sys

import mne
import numpy as np
import pandas as pd

RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
         "P4", "P8", "O1", "O2"]


def clean(ch):
    c = re.sub(r"^(EEG\s*)", "", ch.strip(), flags=re.I)
    c = re.split(r"[-_ ]", c)[0]
    c = {k.upper(): v for k, v in RENAME.items()}.get(c.upper(), c)
    for s in STD19:
        if s.upper() == c.upper():
            return s
    return c


def read_std(path, fs_out):
    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    names = {clean(c): c for c in raw.ch_names}
    have = [c for c in STD19 if c in names]
    raw.pick([names[c] for c in have])
    raw.rename_channels({names[c]: c for c in have})
    if fs_out and int(round(raw.info["sfreq"])) != fs_out:
        raw.resample(fs_out, verbose="ERROR")
    return (raw.get_data() * 1e6).astype(np.float32), have, int(round(raw.info["sfreq"]))


def save(out, name, rows, channels, fs):
    pd.DataFrame(rows).to_csv(f"{out}/{name}/participants.csv", index=False)
    json.dump({"channels": channels, "fs": fs}, open(f"{out}/{name}/info.json", "w"))
    d = pd.DataFrame(rows)
    print(name, len(d), d["status"].value_counts().to_dict(), "fs", fs, "ch", len(channels), flush=True)


def mumtaz(base):
    out = f"{base}/mumtaz"
    os.makedirs(out, exist_ok=True)
    rows, chans = [], None
    files = glob.glob(f"{base}/raw/mumtaz/*.edf")
    subjects = sorted({re.match(r"(H|MDD)\s*S(\d+)", os.path.basename(f)).group(0).replace(" ", "")
                       for f in files if re.match(r"(H|MDD)\s*S(\d+)", os.path.basename(f))})
    for s in subjects:
        g, n = re.match(r"(H|MDD)S(\d+)", s).groups()
        arrs = {}
        for cond in ("EC", "EO"):
            f = [x for x in files if re.match(rf"{g}\s*S{n}\s+{cond}\.edf$", os.path.basename(x))]
            if f:
                try:
                    x, ch, fs = read_std(f[0], 250)
                except Exception as e:                    # corrupt file in the source dataset
                    print("skip unreadable", os.path.basename(f[0]), type(e).__name__, flush=True)
                    continue
                if len(ch) == 19:
                    arrs[cond], chans = x, ch
        if arrs:
            np.savez(f"{out}/{s}.npz", **arrs)
            rows.append({"subject": s, "status": "MDD" if g == "MDD" else "HEALTHY", "age": np.nan, "sex": "NA"})
    save(base, "mumtaz", rows, chans, 250)


def repod(base):
    out = f"{base}/repod"
    os.makedirs(out, exist_ok=True)
    rows, chans = [], None
    for f in sorted(glob.glob(f"{base}/raw/repod/*.edf")):
        s = os.path.basename(f)[:-4]
        x, ch, fs = read_std(f, 250)
        np.savez(f"{out}/{s}.npz", EC=x)
        chans = ch
        rows.append({"subject": s, "status": "SZ" if s.startswith("s") else "HEALTHY", "age": np.nan, "sex": "NA"})
    save(base, "repod", rows, chans, 250)


MSU_CH = ["F7", "F3", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def msu(base):
    out = f"{base}/msu"
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(f"{base}/raw/msu/**/*.eea", recursive=True)):
        v = np.loadtxt(f, dtype=np.float32)
        x = v.reshape(16, -1)                                # channel-major: 16 x 7680 (60 s @ 128 Hz)
        grp = "SZ" if "sch" in f.lower() else "HEALTHY"
        s = f"{grp[:2]}_{os.path.basename(f)[:-4]}"
        np.savez(f"{out}/{s}.npz", EC=x)
        rows.append({"subject": s, "status": grp, "age": np.nan, "sex": "M"})
    save(base, "msu", rows, MSU_CH, 128)


def aseeg(base):
    """ASEEG ships PREPROCESSED data (1-45 Hz Butterworth, 50 Hz notch, 500 -> 128 Hz; no ICA/ASR) as
    MATLAB v7.3 files EC|EO/normal|sz/<n|sz><k>.mat with 'preprocessed_data' (samples x 19). The
    channel order is not documented; data/aseeg_channel_order.py infers it by matching the group
    inter-channel correlation structure to labelled RepOD recordings (standard row-wise 10-20,
    r = 0.855; all other candidate orders r < 0.1)."""
    import h5py
    out = f"{base}/aseeg"
    os.makedirs(out, exist_ok=True)
    order = json.load(open(f"{base}/aseeg_channel_order.json"))["order"]
    perm = [order.index(c) for c in STD19]
    rows = []
    subs = {}
    for f in glob.glob(f"{base}/raw/aseeg/*/*/*.mat"):
        cond, grp, name = f.split("/")[-3:]
        subs.setdefault((grp, name[:-4]), {})[cond] = f
    for (grp, sid), conds in sorted(subs.items()):
        arrs = {}
        for cond, f in conds.items():
            with h5py.File(f) as h:
                x = np.asarray(h["preprocessed_data"], dtype=np.float32).T   # 19 x samples
            arrs[cond] = x[perm]
        np.savez(f"{out}/{sid}.npz", **arrs)
        rows.append({"subject": sid, "status": "SZ" if grp == "sz" else "HEALTHY", "age": np.nan, "sex": "NA"})
    save(base, "aseeg", rows, STD19, 128)


if __name__ == "__main__":
    base = sys.argv[1]
    for name in (sys.argv[2:] or ["mumtaz", "repod", "msu"]):
        globals()[name](base)

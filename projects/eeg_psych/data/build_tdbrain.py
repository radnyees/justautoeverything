"""
projects/eeg_psych/data/build_tdbrain.py -- TDBRAIN V3.1 (van Dijk et al. 2022, Sci Data;
brainclinics.com/resources; CC BY 4.0 under the TDBRAIN DUA -- do NOT redistribute) ->
/data/eeg_psych/tdbrain/<sub>.npz (EC, EO: float32 microvolts, 26 EEG channels, 250 Hz) +
participants.csv for every DISCOVERY subject with formal_status MDD / ADHD / ADD / HEALTHY
(first session only).
    python projects/eeg_psych/data/build_tdbrain.py /data/eeg_psych/tdbrain PASSWORD   (or $TDBRAIN_PW)
"""
import json
import os
import subprocess
import sys
import tempfile

import mne
import numpy as np
import pandas as pd

ZIP = "TDBRAIN_Dataset_V3_1_Encr.zip"
FS = 250
EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
       "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]


def main(out):
    pw = sys.argv[2] if len(sys.argv) > 2 else os.environ["TDBRAIN_PW"]
    p = pd.read_excel(f"{out}/TDBRAIN_participants_V3.1.xlsx")
    p = p[p["formal_status"].isin(["MDD", "ADHD", "ADD", "HEALTHY"])]
    p = p.sort_values("sessID").drop_duplicates("TDBRAIN_ID")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        members = []
        for sid, ses in zip(p["TDBRAIN_ID"], p["sessID"]):
            for cond in ("EC", "EO"):
                members.append(f"TDBRAIN_Dataset_V3_1/{sid}/ses-{int(ses)}/eeg/{sid}_ses-{int(ses)}_task-rest{cond}_eeg.bdf")
        lst = f"{tmp}/members.txt"
        open(lst, "w").write("\n".join(members) + "\n")
        r = subprocess.run(["bsdtar", "-xf", f"{out}/{ZIP}", "--passphrase", pw, "-C", tmp, "-T", lst],
                           capture_output=True, text=True)
        print("bsdtar rc", r.returncode, r.stderr[-500:], flush=True)
        for _, row in p.iterrows():
            sid, ses = row["TDBRAIN_ID"], int(row["sessID"])
            arrs = {}
            for cond in ("EC", "EO"):
                f = f"{tmp}/TDBRAIN_Dataset_V3_1/{sid}/ses-{ses}/eeg/{sid}_ses-{ses}_task-rest{cond}_eeg.bdf"
                if not os.path.exists(f):
                    continue
                raw = mne.io.read_raw_bdf(f, preload=True, verbose="ERROR")
                raw.pick([c for c in EEG if c in raw.ch_names])
                if len(raw.ch_names) != len(EEG):
                    continue
                raw.reorder_channels(EEG)
                raw.resample(FS, verbose="ERROR")
                arrs[cond] = (raw.get_data() * 1e6).astype(np.float32)
            if not arrs:
                continue
            np.savez(f"{out}/{sid}.npz", **arrs)
            status = "ADHD" if row["formal_status"] in ("ADHD", "ADD") else row["formal_status"]
            rows.append({"subject": sid, "status": status, "age": row["age"],
                         "sex": {1: "M", 0: "F"}.get(row["gender"], "NA"),
                         "has_EC": "EC" in arrs, "has_EO": "EO" in arrs})
    pd.DataFrame(rows).to_csv(f"{out}/participants.csv", index=False)
    json.dump({"channels": EEG, "fs": FS}, open(f"{out}/info.json", "w"))
    d = pd.DataFrame(rows)
    print(d["status"].value_counts().to_dict())
    print(d.groupby("status")["age"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])

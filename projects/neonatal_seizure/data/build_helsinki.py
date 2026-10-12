"""
projects/neonatal_seizure/data/build_helsinki.py -- Helsinki neonatal EEG (Stevenson et al. 2019,
Sci Data; Zenodo 2547147, CC BY 4.0) -> /data/neonatal/.

79 term neonates, 19 scalp electrodes (10-20 referential), 256 Hz, ~74 min median. Three experts
(A, B, C) annotated seizures per second. Label used here: majority vote (>= 2 of 3 experts).
    python projects/neonatal_seizure/data/build_helsinki.py /data/neonatal
Writes eeg<k>.npy (float32, channels x samples, microvolts), labels.npz (per-second A/B/C/majority
for each recording), channels.json and clinical_information.csv.
"""
import io
import json
import os
import sys
import tempfile
import urllib.request

import mne
import numpy as np
import pandas as pd

Z = "https://zenodo.org/api/records/2547147/files/{}/content"
RAW = ""                                            # set from argv[2]: directory with the raw files
EEG_CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
          "T5", "T6", "Fz", "Cz", "Pz"]


def get(name, tries=6):
    local = os.path.join(RAW, name)                 # pre-uploaded copy (Zenodo blocks some clouds)
    if RAW and os.path.exists(local):
        return open(local, "rb").read()
    for t in range(tries):
        try:
            return urllib.request.urlopen(Z.format(name), timeout=300).read()
        except Exception as e:                      # Zenodo drops long downloads now and then
            print(f"retry {name} ({t + 1}): {e}", flush=True)
    raise RuntimeError(f"could not download {name}")


def main(out):
    os.makedirs(out, exist_ok=True)
    open(f"{out}/clinical_information.csv", "wb").write(get("clinical_information.csv"))
    ann = {k: pd.read_csv(io.BytesIO(get(f"annotations_2017_{k}.csv"))) for k in "ABC"}
    labels = {}
    for i in range(1, 80):
        if os.path.exists(f"{out}/eeg{i}.npy"):              # resume: already converted
            x = np.load(f"{out}/eeg{i}.npy", mmap_mode="r")
        else:
            x = None
        if x is None:
          with tempfile.NamedTemporaryFile(suffix=".edf") as tf:
            tf.write(get(f"eeg{i}.edf"))
            tf.flush()
            raw = mne.io.read_raw_edf(tf.name, preload=True, verbose="ERROR")
          names = {c.upper().replace("EEG ", "").replace("-REF", "").strip(): c for c in raw.ch_names}
          picks = [names[c.upper()] for c in EEG_CH]
          x = raw.get_data(picks=picks) * 1e6                   # volts -> microvolts
          assert int(raw.info["sfreq"]) == 256, raw.info["sfreq"]
          np.save(f"{out}/eeg{i}.npy", x.astype(np.float32))
        nsec = x.shape[1] // 256
        per = {k: ann[k][str(i)].dropna().values[:nsec].astype(np.int8) for k in "ABC"}
        n = min(nsec, *(len(v) for v in per.values()))
        maj = ((per["A"][:n] + per["B"][:n] + per["C"][:n]) >= 2).astype(np.int8)
        labels[f"eeg{i}"] = {"A": per["A"][:n], "B": per["B"][:n], "C": per["C"][:n], "majority": maj}
        print(f"eeg{i}", x.shape, "sec", n, "seizure sec", int(maj.sum()), flush=True)
    np.savez(f"{out}/labels.npz", **{f"{r}|{k}": v for r, d in labels.items() for k, v in d.items()})
    json.dump(EEG_CH, open(f"{out}/channels.json", "w"))


if __name__ == "__main__":
    RAW = sys.argv[2] if len(sys.argv) > 2 else ""
    main(sys.argv[1])

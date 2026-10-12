"""
projects/neonatal_seizure/prepare.py -- FIXED data/label/metric definition (never edited by the loop).

Task
  helsinki : per-second seizure detection in the Helsinki neonatal EEG dataset (Stevenson et al.
             2019; 79 term neonates, 19 scalp electrodes, 256 Hz). Label = majority vote (>= 2 of
             3 experts) per second. Splits are grouped by neonate and stratified by whether the
             neonate had any seizure.
Metric: AUROC over all seconds of the held-out neonates concatenated (the field's "concatenated
AUC"). items = recording ids; ctx["load_eeg"](items) -> float32 (19 x samples, microvolts);
predictions = one float array per recording with ctx["n_seconds"](item) values (one per second).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "neonatal"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]


@lru_cache(maxsize=1)
def _labels():
    z = np.load(DATA / "labels.npz")
    return {k.split("|")[0]: z[k] for k in z.files if k.endswith("|majority")}


def load_eeg(items):
    return [np.load(DATA / f"{i}.npy", mmap_mode="r") for i in items]


def load(task):
    if task != "helsinki":
        raise KeyError(task)
    lab = _labels()
    items = sorted(lab, key=lambda s: int(s[3:]))
    y = [lab[i].astype(np.int8) for i in items]
    strata = np.array([str(int(v.any())) for v in y])
    nsec = {i: len(lab[i]) for i in items}
    ctx = {"fs": 256, "channels": json.loads((DATA / "channels.json").read_text()),
           "load_eeg": load_eeg, "n_seconds": lambda item: nsec[item]}
    return {"items": items, "y": y, "groups": np.array(items), "strata": strata, "meta": None,
            "ctx": ctx, "external": {}}


def score(y_true, y_pred, task):
    yt = np.concatenate([np.asarray(a).ravel() for a in y_true])
    yp = np.concatenate([np.asarray(a, dtype=float).ravel() for a in y_pred])
    if yt.shape != yp.shape:
        raise ValueError(f"prediction length {yp.shape} != label length {yt.shape}")
    return float(roc_auc_score(yt, yp))

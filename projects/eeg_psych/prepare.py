"""
projects/eeg_psych/prepare.py -- FIXED data/label/metric definition (never edited by the loop).

Tasks (resting-state EEG, subject-level classification)
  mdd_tdbrain  : major depression (formal dx) vs healthy, TDBRAIN V3.1; eyes closed + eyes open;
                 19 standard 10-20 channels shared with Mumtaz; 250 Hz. AGE-MATCHED: both groups
                 restricted to the overlapping age range (age is never given to the model).
                 External (scored once): Mumtaz 2016 MDD vs healthy (different site, amplifier).
  adhd_tdbrain : ADHD/ADD (formal dx) vs healthy, TDBRAIN V3.1; EC + EO; 26 channels; 250 Hz;
                 AGE-MATCHED as above.
  sz_aseeg     : schizophrenia vs healthy, ASEEG; eyes closed; the 16 channels shared with the
                 external sets; 128 Hz. External: MSU adolescents (16 ch, 1 min EC), RepOD (EC).
  abnormal_tuab: abnormal vs normal clinical EEG (TUH Abnormal v3.0.1, official TRAIN split,
                 grouped by patient); 5 min after the first minute, 19 channels, 100 Hz; condition
                 "REST". External: the official TUAB EVAL split (276 recordings), scored once.
items = list of "dataset:subject" ids; ctx["load_eeg"](items, "EC"|"EO") returns a list of float32
arrays (channels x samples, microvolts) at ctx["fs"] with rows in ctx["channels"] order
(None where a subject lacks that condition).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "eeg_psych"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
        "P4", "P8", "O1", "O2"]
CH16 = ["F7", "F3", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
SPEC = {"mdd_tdbrain": dict(fs=250, channels=CH19, conds=("EC", "EO")),
        "adhd_tdbrain": dict(fs=250, channels=None, conds=("EC", "EO")),
        "sz_aseeg": dict(fs=128, channels=CH16, conds=("EC",)),
        "abnormal_tuab": dict(fs=100, channels=CH19, conds=("REST",))}


@lru_cache(maxsize=None)
def _info(dataset):
    return json.loads((DATA / dataset / "info.json").read_text())


@lru_cache(maxsize=None)
def _packed(dataset, cond):
    """Memory-mapped packed corpus (TUAB): (array, {subject: [row, n_samples]}) or None."""
    f = DATA / dataset / f"{cond}.npy"
    if not f.exists():
        return None
    return np.load(f, mmap_mode="r"), json.loads((DATA / dataset / f"{cond}_index.json").read_text())


@lru_cache(maxsize=None)
def _raw(item, cond):
    dataset, subj = item.split(":", 1)
    pk = _packed(dataset, cond)
    if pk is not None:
        row, n = pk[1][subj]
        return np.asarray(pk[0][row, :, :n])
    z = np.load(DATA / dataset / f"{subj}.npz")
    return z[cond] if cond in z.files else None


def _loader(task):
    spec = SPEC[task]

    def load_eeg(items, cond="EC"):
        from mne.filter import resample
        out = []
        for it in items:
            x = _raw(it, cond)
            if x is None:
                out.append(None)
                continue
            info = _info(it.split(":", 1)[0])
            chans = spec["channels"] or info["channels"]
            idx = [info["channels"].index(c) for c in chans]
            x = x[idx]
            if info["fs"] != spec["fs"]:
                x = resample(x.astype(np.float64), up=spec["fs"], down=info["fs"], axis=-1).astype(np.float32)
            out.append(x)
        return out
    return load_eeg


def _age_matched(p, case):
    a = p[p["status"] == case]["age"].dropna()
    h = p[p["status"] == "HEALTHY"]["age"].dropna()
    lo, hi = max(a.quantile(0.05), h.quantile(0.05)), min(a.quantile(0.95), h.quantile(0.95))
    return p[p["status"].isin([case, "HEALTHY"]) & p["age"].between(lo, hi)], (round(lo, 1), round(hi, 1))


def _subjects(dataset, statuses):
    p = pd.read_csv(DATA / dataset / "participants.csv")
    return p[p["status"].isin(statuses)]


def load(task):
    spec = SPEC[task]
    external = {}
    if task in ("mdd_tdbrain", "adhd_tdbrain"):
        case = "MDD" if task == "mdd_tdbrain" else "ADHD"
        p, age_range = _age_matched(pd.read_csv(DATA / "tdbrain" / "participants.csv"), case)
        p = p[p["has_EC"]]
        items = [f"tdbrain:{s}" for s in p["subject"]]
        y = (p["status"] == case).astype(int).values
        meta = pd.DataFrame(index=items)                   # v2: no demographics
        note = {"age_matched_range": age_range}
        if task == "mdd_tdbrain":
            m = _subjects("mumtaz", ["MDD", "HEALTHY"])
            ext_items = [f"mumtaz:{s}" for s in m["subject"]]
            external["mumtaz"] = {"items": ext_items, "y": (m["status"] == "MDD").astype(int).values,
                                  "groups": np.array(ext_items), "meta": pd.DataFrame(index=ext_items)}
    elif task == "sz_aseeg":
        p = _subjects("aseeg", ["SZ", "HEALTHY"])
        items = [f"aseeg:{s}" for s in p["subject"]]
        y = (p["status"] == "SZ").astype(int).values
        meta = pd.DataFrame(index=items)                   # v2: no demographics
        note = {}
        for ds in ("msu", "repod"):
            m = _subjects(ds, ["SZ", "HEALTHY"])
            ext_items = [f"{ds}:{s}" for s in m["subject"]]
            external[ds] = {"items": ext_items, "y": (m["status"] == "SZ").astype(int).values,
                            "groups": np.array(ext_items), "meta": pd.DataFrame(index=ext_items)}
    elif task == "abnormal_tuab":
        p = pd.read_csv(DATA / "tuab" / "participants.csv")
        tr, ev = p[p["split"] == "train"], p[p["split"] == "eval"]
        items = [f"tuab:{s}" for s in tr["subject"]]
        y = (tr["status"] == "abnormal").astype(int).values
        meta = pd.DataFrame(index=items)
        note = {}
        groups = np.array([f"tuab:{g}" for g in tr["patient"]])
        ext_items = [f"tuab:{s}" for s in ev["subject"]]
        external["tuab_official_eval"] = {
            "items": ext_items, "y": (ev["status"] == "abnormal").astype(int).values,
            "groups": np.array([f"tuab:{g}" for g in ev["patient"]]), "meta": pd.DataFrame(index=ext_items)}
        ctx = {"fs": spec["fs"], "channels": spec["channels"], "conditions": list(spec["conds"]),
               "load_eeg": _loader(task)}
        return {"items": items, "y": y, "groups": groups, "strata": y.astype(str), "meta": meta,
                "ctx": ctx, "external": external}
    else:
        raise KeyError(task)
    chans = spec["channels"] or _info("tdbrain")["channels"]
    ctx = {"fs": spec["fs"], "channels": chans, "conditions": list(spec["conds"]),
           "load_eeg": _loader(task), **note}
    return {"items": items, "y": y, "groups": np.array(items), "strata": y.astype(str), "meta": meta,
            "ctx": ctx, "external": external}


def score(y_true, y_pred, task):
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_pred, dtype=float)))

"""
core/confounds_data.py -- collect per-item demographics and IBD sample metadata (runs on Modal: /data).

Writes /data/analysis/confounds.json (NOT committed: per-subject demographics, TDBRAIN DUA):
  demographics[task][item] = {"age": float|None, "sex": "M"|"F"|None, "site": str|None}
  ibd[study] = metadata summary (columns, sample sites, samples per subject)
Used by core/confounds.py, which writes only aggregate results to the repo.
    ./core/run.sh python core/prep.py core/confounds_data.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data")


def prepare(project):
    spec = importlib.util.spec_from_file_location("prep_" + project, ROOT / "projects" / project / "prepare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sex(v):
    s = str(v).strip().upper()
    return {"M": "M", "MALE": "M", "1": "M", "1.0": "M", "F": "F", "FEMALE": "F", "0": "F", "0.0": "F",
            "2": "F", "2.0": "F"}.get(s)


def items_of(ds):
    it = ds["items"]
    return [str(i) for i in (it.index if hasattr(it, "iloc") else it)]


def main():
    out = {"demographics": {}, "ibd": {}, "notes": {}}
    # fMRI: ADHD-200 (Gender 1 = male, 0 = female), ABIDE (SEX 1 = male, 2 = female)
    fm = prepare("fmri_connectivity")
    ph = pd.read_csv(DATA / "fmri" / "adhd200" / "phenotypes.tsv", sep="\t")
    ph["sid"] = ph["ScanDir ID"].astype(str).str.zfill(7)
    ph = ph.drop_duplicates("sid").set_index("sid")
    ds = fm.load("adhd_adhd200")
    def add_y(task, ds):
        for i, yy in zip(items_of(ds), np.asarray(ds["y"]).tolist()):
            out["demographics"][task][i]["y"] = int(yy)

    out["demographics"]["adhd_adhd200"] = {
        i: {"age": float(ph.loc[i, "Age"]) if pd.notna(ph.loc[i, "Age"]) else None,
            "sex": {1: "M", 0: "F"}.get(int(ph.loc[i, "Gender"])) if pd.notna(ph.loc[i, "Gender"]) else None,
            "site": str(s)} for i, s in zip(items_of(ds), ds["sites"])}
    add_y("adhd_adhd200", ds)
    ph = pd.read_csv(DATA / "fmri" / "abide" / "phenotypes.csv").drop_duplicates("FILE_ID")
    ph.index = ph["FILE_ID"].astype(str)
    ds = fm.load("autism_abide")
    out["demographics"]["autism_abide"] = {
        i: {"age": float(ph.loc[i, "AGE_AT_SCAN"]), "sex": {1: "M", 2: "F"}.get(int(ph.loc[i, "SEX"])),
            "site": str(s)} for i, s in zip(items_of(ds), ds["sites"])}
    add_y("autism_abide", ds)
    # EEG
    ee = prepare("eeg_psych")
    for task, dataset in [("mdd_tdbrain", "tdbrain"), ("adhd_tdbrain", "tdbrain"), ("sz_aseeg", "aseeg"),
                          ("abnormal_tuab", "tuab")]:
        p = pd.read_csv(DATA / "eeg_psych" / dataset / "participants.csv")
        out["notes"][f"{dataset}_columns"] = list(p.columns)
        p.index = [f"{dataset}:{s}" for s in p["subject"]]
        ds = ee.load(task)
        d = {}
        for i in items_of(ds):
            row = p.loc[i] if i in p.index else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            age = float(row["age"]) if row is not None and "age" in p and pd.notna(row["age"]) else None
            sex = _sex(row["sex"]) if row is not None and "sex" in p else None
            d[i] = {"age": age, "sex": sex, "site": None}
        out["demographics"][task] = d
        add_y(task, ds)
        out["notes"][f"{task}_n_with_age"] = int(sum(v["age"] is not None for v in d.values()))
        out["notes"][f"{task}_n_with_sex"] = int(sum(v["sex"] is not None for v in d.values()))
    # IBD metadata
    for st in ["ibd_gevers_2014", "ibd_alm", "ibd_engstrand_maxee", "ibd_huttenhower"]:
        m = pd.read_parquet(DATA / "microbiome" / f"{st}.meta.parquet")
        keep = m["DiseaseState"].isin(["CD", "UC", "H", "nonIBD"])
        m = m[keep]
        info = {"columns": list(m.columns), "n_samples": int(len(m)),
                "disease": m["DiseaseState"].value_counts().to_dict()}
        for c in m.columns:
            if c.lower() in ("sample_location", "body_site", "site", "tissue", "sample_type", "location"):
                info[f"values:{c}"] = m[c].astype(str).value_counts().to_dict()
        subj_cols = [c for c in m.columns if "subject" in c.lower() or "patient" in c.lower() or "host" in c.lower()]
        info["subject_columns"] = subj_cols
        for c in subj_cols:
            vc = m[c].astype(str).value_counts()
            info[f"samples_per_{c}"] = {"n_unique": int(vc.size), "max": int(vc.max()), "n_with_gt1": int((vc > 1).sum())}
        info["index_example"] = [str(x) for x in m.index[:5]]
        info["index_all"] = [str(x) for x in m.index]
        info["disease_all"] = m["DiseaseState"].astype(str).tolist()
        for c in m.columns:                       # every low-cardinality column (sample type, tissue, ...)
            vc = m[c].astype(str).value_counts()
            if 1 < vc.size <= 12 and c != "DiseaseState":
                info[f"values:{c}"] = vc.to_dict()
        out["ibd"][st] = info
        print(st, json.dumps({k: v for k, v in info.items() if k.startswith("values:")}, default=str)[:1500], flush=True)
    print(json.dumps(out["notes"]), flush=True)
    (DATA / "analysis").mkdir(exist_ok=True)
    (DATA / "analysis" / "confounds.json").write_text(json.dumps(out, default=str))
    print("wrote /data/analysis/confounds.json", flush=True)


if __name__ == "__main__":
    main()

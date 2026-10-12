"""
projects/fmri_connectivity/prepare.py -- FIXED data/label/metric definition (never edited by the loop).

Tasks
  autism_abide : ABIDE I Preprocessed (PCP, CPAC), autism (DX_GROUP 1) vs typically developing (2),
                 ~1,035 subjects, 20 sites. Time series for 7 atlases x {filt_noglobal, filt_global}.
  adhd_adhd200 : ADHD-200 Preprocessed (Athena), ADHD (any subtype, DX 1/2/3) vs typically
                 developing (DX 0), subjects passing Athena QC with a known diagnosis, 8 sites.
items = list of subject ids; ctx["load_timeseries"](items, atlas, strategy) returns their ROI
time series. Splits are grouped by subject (one scan each) and stratified by label x site.
"""
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "fmri"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]
ABIDE_ATLASES = ["aal", "cc200", "cc400", "dosenbach160", "ez", "ho", "tt"]
ADHD_ATLASES = ["aal", "cc200", "cc400", "ez", "ho", "tt"]
ADHD_SITES = {1: "Peking", 2: "Brown", 3: "KKI", 4: "NeuroIMAGE", 5: "NYU", 6: "OHSU",
              7: "Pittsburgh", 8: "WashU"}


@lru_cache(maxsize=None)
def _ts_table(dataset, strategy, atlas):
    with open(DATA / dataset / strategy / f"{atlas}.pkl", "rb") as f:
        return pickle.load(f)


def _loader(dataset):
    def load_timeseries(items, atlas, strategy):
        tab = _ts_table(dataset, strategy, atlas)
        return [tab[str(i)] for i in items]
    return load_timeseries


def load(task):
    if task == "autism_abide":
        ph = pd.read_csv(DATA / "abide" / "phenotypes.csv")
        have = set(_ts_table("abide", "filt_noglobal", "cc200"))
        ph = ph[ph["FILE_ID"].astype(str).isin(have)]
        for st in ["filt_noglobal", "filt_global"]:
            for at in ABIDE_ATLASES:
                ph = ph[ph["FILE_ID"].astype(str).isin(set(_ts_table("abide", st, at)))]
        items = ph["FILE_ID"].astype(str).tolist()
        y = (ph["DX_GROUP"] == 1).astype(int).values
        meta = pd.DataFrame({"site": ph["SITE_ID"].astype(str).values,
                             "age": ph["AGE_AT_SCAN"].astype(float).values,
                             "sex": ph["SEX"].map({1: "M", 2: "F"}).values,
                             "mean_fd": pd.to_numeric(ph["func_mean_fd"], errors="coerce").values},
                            index=items)
        ctx = {"dataset": "ABIDE I (PCP, CPAC)", "atlases": ABIDE_ATLASES,
               "strategies": ["filt_noglobal", "filt_global"], "load_timeseries": _loader("abide")}
    elif task == "adhd_adhd200":
        ph = pd.read_csv(DATA / "adhd200" / "phenotypes.tsv", sep="\t")
        ph["sid"] = ph["ScanDir ID"].astype(str).str.zfill(7)
        ph = ph[ph["DX"].astype(str).isin(["0", "1", "2", "3"]) & (ph["QC_Athena"] == 1)]
        for at in ADHD_ATLASES:
            ph = ph[ph["sid"].isin(set(_ts_table("adhd200", "athena", at)))]
        ph = ph.drop_duplicates("sid")
        items = ph["sid"].tolist()
        y = (ph["DX"].astype(str) != "0").astype(int).values
        meta = pd.DataFrame({"site": ph["Site"].map(ADHD_SITES).values,
                             "age": pd.to_numeric(ph["Age"], errors="coerce").values,
                             "sex": ph["Gender"].map({0: "F", 1: "M"}).values}, index=items)
        ctx = {"dataset": "ADHD-200 (Athena)", "atlases": ADHD_ATLASES, "strategies": ["athena"],
               "load_timeseries": _loader("adhd200")}
    else:
        raise KeyError(task)
    strata = np.array([f"{a}|{b}" for a, b in zip(y, meta["site"])])
    return {"items": items, "y": y, "groups": np.array(items), "strata": strata, "meta": meta,
            "ctx": ctx, "external": {}}


def score(y_true, y_pred, task):
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_pred, dtype=float)))

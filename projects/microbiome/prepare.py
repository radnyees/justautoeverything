"""
projects/microbiome/prepare.py -- FIXED data/label/metric definition for the microbiome tasks
(the loop never edits this). Data: MicrobiomeHD (Duvallet et al. 2017), built by data/build.py.

Tasks
  crc_baxter : colorectal cancer vs healthy, Baxter 2016 stool 16S (one sample per subject).
               External (scored once at the end): Zeller, Zackular, Xiang, Zhao (CRC vs H).
  ibd_pooled : inflammatory bowel disease (Crohn's or ulcerative colitis) vs controls, POOLED over
               four stool 16S studies (Gevers 2014 RISK, Alm, Engstrand, Huttenhower). OTU tables
               are merged on the RDP taxonomy string (OTU ids are study-specific), so columns
               are taxonomy lineages without the d__denovoN suffix. Split/strata by study+label.
               (Gevers alone has only 16 stool controls -- too few to evaluate; see
               archive/ibd_gevers_stool_only/.)
crc items = DataFrame of RAW OTU counts (samples x OTUs); columns are RDP taxonomy strings
'k__..;p__..;c__..;o__..;f__..;g__..;s__..;d__denovoN'.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "microbiome"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]
CONTROL = {"H", "nonIBD", "nonCRC_H"}


def _study(name):
    c = pd.read_parquet(DATA / f"{name}.counts.parquet")
    m = pd.read_parquet(DATA / f"{name}.meta.parquet")
    return c.astype(np.int32), m


def _binary(name, case, controls):
    c, m = _study(name)
    keep = m["DiseaseState"].isin([case, *controls]).values
    c, m = c.loc[keep], m.loc[keep]
    y = (m["DiseaseState"] == case).astype(int).values
    loc = m["sample_location"] if "sample_location" in m else pd.Series("stool", index=m.index)
    meta = pd.DataFrame({"sample_location": loc.values, "study": name}, index=m.index)
    groups = (m["subject"] if "subject" in m else pd.Series(m.index, index=m.index)).astype(str).values
    return c, y, groups, meta


def load(task):
    if task == "crc_baxter":
        c, y, g, meta = _binary("crc_baxter", "CRC", ["H"])
        ext_names = ["crc_zeller", "crc_zackular", "crc_xiang", "crc_zhao"]
        ext_args = ("CRC", ["H"])
    elif task == "ibd_pooled":
        parts = []
        for st in ["ibd_gevers_2014", "ibd_alm", "ibd_engstrand_maxee", "ibd_huttenhower"]:
            c, m = _study(st)
            keep = m["DiseaseState"].isin(["CD", "UC", "H", "nonIBD"]).values
            c, m = c.loc[keep], m.loc[keep]
            c.columns = [col.rsplit(";d__", 1)[0] for col in c.columns]   # drop study-specific OTU id
            c = c.T.groupby(level=0).sum().T
            c.index = [f"{st}:{i}" for i in c.index]
            subj = (m["subject"] if "subject" in m else pd.Series(m.index, index=m.index)).astype(str)
            parts.append((c, (m["DiseaseState"].isin(["CD", "UC"])).astype(int).values,
                          np.array([f"{st}:{x}" for x in subj]), st))
        c = pd.concat([p[0] for p in parts], axis=0).fillna(0).astype(np.int32)
        y = np.concatenate([p[1] for p in parts])
        g = np.concatenate([p[2] for p in parts])
        study = np.concatenate([[p[3]] * len(p[1]) for p in parts])
        meta = pd.DataFrame({"sample_location": "stool", "study": study}, index=c.index)
        strata = np.array([f"{a}|{b}" for a, b in zip(y, study)])
        return {"items": c, "y": y, "groups": g, "strata": strata, "meta": meta,
                "ctx": {"study": task, "taxonomy_format":
                        "k__;p__;c__;o__;f__;g__;s__ (RDP lineage; OTUs merged by lineage across studies)"},
                "external": {}}
    else:
        raise KeyError(task)
    strata = np.array([f"{a}|{b}" for a, b in zip(y, meta["sample_location"])])
    external = {}
    for n in ext_names:
        ec, ey, eg, em = _binary(n, *ext_args)
        if len(set(ey)) == 2:
            external[n] = {"items": ec, "y": ey, "groups": eg, "meta": em}
    return {"items": c, "y": y, "groups": g, "strata": strata, "meta": meta,
            "ctx": {"study": task, "taxonomy_format":
                    "k__;p__;c__;o__;f__;g__;s__;d__denovoN (RDP, de novo 100% OTUs)"},
            "external": external}


def score(y_true, y_pred, task):
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_pred, dtype=float)))

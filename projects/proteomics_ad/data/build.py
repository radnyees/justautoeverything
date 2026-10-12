"""
projects/proteomics_ad/data/build.py -- open Alzheimer's proteomics datasets -> parquet.

  plasma : PXD065835 / iProX IPX0012494000 -- plasma DIA (Orbitrap Astral, DIA-NN pg_matrix),
           53 AD + 36 cognitively normal controls (class encoded in run names; QC runs dropped).
  csf    : PXD016278 (Bader et al. 2020, Mol Syst Biol) -- CSF DIA (Spectronaut), 4 cohorts,
           with clinical AD diagnosis, biochemical AD classification, age, sex, site.

    python projects/proteomics_ad/data/build.py OUT_DIR
"""
import io
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

PLASMA = "http://download.iprox.org/IPX0012494000/IPX0012494002/YAS202404020004-1.pg_matrix.tsv"
BADER = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2020/05/PXD016278/"


def get(url):
    return urllib.request.urlopen(url, timeout=900).read()


def main(out):
    os.makedirs(out, exist_ok=True)
    # ---- plasma
    pg = pd.read_csv(io.BytesIO(get(PLASMA)), sep="\t")
    runs = [c for c in pg.columns if c.endswith(".raw")]
    names = pg["Genes"].fillna("").astype(str) + "|" + pg["Protein.Group"].astype(str)
    X = pg[runs].T
    X.columns = names.values
    info = []
    for r in runs:
        m = re.search(r"-(AD|NC|QC)(\d+)\.raw$", r)
        batch = "B2" if "FUCE" in r else "B1"
        info.append((m.group(1), m.group(1) + m.group(2), batch))
    meta = pd.DataFrame(info, columns=["group", "sample", "batch"], index=X.index)
    keep = meta["group"].isin(["AD", "NC"]).values
    X, meta = X.loc[keep], meta.loc[keep]
    X.index = meta["sample"].values
    meta.index = meta["sample"].values
    X.astype("float64").to_parquet(f"{out}/plasma.X.parquet")
    meta.to_parquet(f"{out}/plasma.meta.parquet")
    print("plasma", X.shape, meta["group"].value_counts().to_dict(), meta["batch"].value_counts().to_dict())

    # ---- CSF (Bader 2020)
    z = zipfile.ZipFile(io.BytesIO(get(BADER + "proteingroupsreport_usedforanalysis.zip")))
    f = next(n for n in z.namelist() if n.endswith("analysis.csv"))
    d = pd.read_csv(z.open(f))
    qcols = [c for c in d.columns if c.endswith("PG.Quantity")]
    names = d["PG.Genes"].fillna("").astype(str) + "|" + d["PG.ProteinGroups"].astype(str)
    Xc = d[qcols].T
    Xc.columns = names.values
    Xc.index = [re.sub(r"^\[\d+\]\s*", "", c) for c in Xc.index]   # "[12] run.PG.Quantity"
    Xc = Xc.apply(pd.to_numeric, errors="coerce")
    za = zipfile.ZipFile(io.BytesIO(get(BADER + "annotationofsamples_AM1.5.11.zip")))
    a = pd.read_excel(za.open(next(n for n in za.namelist() if n.endswith(".xlsx"))))
    a = a.set_index("sample name")
    common = [c for c in Xc.index if c in a.index]
    Xc, a = Xc.loc[common], a.loc[common]
    excluded = a["comment"].fillna("").str.startswith("excluded").values
    Xc, a = Xc.loc[~excluded], a.loc[~excluded]
    sid = [f"csf{i:03d}" for i in range(len(a))]
    Xc.index, a.index = sid, sid
    meta_c = pd.DataFrame({"site": a["collection site"].astype(str), "age": a["age"],
                           "sex": a["gender"].astype(str),
                           "clinical": a["clinical AD diagnosis"].astype(str),
                           "biochemical": a["biochemical AD classification"].astype(str),
                           "cohort_id": a["cohort sample ID"].astype(str)}, index=sid)
    Xc.astype("float64").to_parquet(f"{out}/csf.X.parquet")
    meta_c.to_parquet(f"{out}/csf.meta.parquet")
    print("csf", Xc.shape, meta_c["clinical"].value_counts().to_dict(), meta_c["site"].value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])

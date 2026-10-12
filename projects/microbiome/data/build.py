"""
projects/microbiome/data/build.py -- MicrobiomeHD (Duvallet et al. 2017, Zenodo 840333, CC-BY-NC)
-> per-study raw OTU count tables (RDP-classified de novo 100% OTUs) + metadata, as parquet.

    python projects/microbiome/data/build.py OUT_DIR      # then upload OUT_DIR to jae-data:/microbiome

Per study writes  <study>.counts.parquet  (samples x OTUs, raw counts; column = full RDP
taxonomy string ending in d__denovoN) and <study>.meta.parquet (DiseaseState + covariates).
"""
import glob
import io
import os
import sys
import tarfile
import urllib.request

import pandas as pd

ZENODO = "https://zenodo.org/api/records/840333/files/{}_results.tar.gz/content"
STUDIES = ["crc_baxter", "crc_zeller", "crc_zackular", "crc_xiang", "crc_zhao",
           "ibd_gevers_2014", "ibd_alm", "ibd_engstrand_maxee", "ibd_huttenhower"]


def main(out):
    os.makedirs(out, exist_ok=True)
    for st in STUDIES:
        raw = urllib.request.urlopen(ZENODO.format(st), timeout=600).read()
        tf = tarfile.open(fileobj=io.BytesIO(raw))
        names = tf.getnames()
        tab = next(n for n in names if n.endswith(".otu_table.100.denovo.rdp_assigned"))
        met = next(n for n in names if n.endswith(".metadata.txt"))
        counts = pd.read_csv(tf.extractfile(tab), sep="\t", index_col=0).T   # samples x OTUs
        meta = pd.read_csv(tf.extractfile(met), sep="\t", index_col=0, encoding="latin-1")
        counts.index = counts.index.astype(str)
        meta.index = meta.index.astype(str)
        common = counts.index.intersection(meta.index)
        counts, meta = counts.loc[common], meta.loc[common]
        counts = counts.loc[:, counts.sum(0) > 0]
        meta = meta.astype(str)
        counts.to_parquet(f"{out}/{st}.counts.parquet")
        meta.to_parquet(f"{out}/{st}.meta.parquet")
        print(st, counts.shape, meta["DiseaseState"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

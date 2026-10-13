"""
projects/fmri_connectivity/data/build_abide.py -- ABIDE I Preprocessed (PCP, CPAC pipeline)
ROI time series for 7 atlases x 2 denoising strategies (band-pass filtered, with / without
global signal regression) + phenotypes -> /data/fmri/abide/.

Source (open S3, CC BY-NC-SA): s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/
    python projects/fmri_connectivity/data/build_abide.py /data/fmri/abide
Writes <strategy>/<atlas>.pkl = {FILE_ID: float32 array (timepoints x ROIs)} and phenotypes.csv.
"""
import io
import os
import pickle
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

S3 = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ATLASES = ["aal", "cc200", "cc400", "dosenbach160", "ez", "ho", "tt"]
STRATEGIES = ["filt_noglobal", "filt_global"]


def fetch(url, tries=5):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if t == tries - 1:
                return None


def main(out):
    os.makedirs(out, exist_ok=True)
    ph = pd.read_csv(io.BytesIO(fetch(f"{S3}/Phenotypic_V1_0b_preprocessed1.csv")))
    ph = ph[ph["FILE_ID"] != "no_filename"].copy()
    ph.to_csv(f"{out}/phenotypes.csv", index=False)
    ids = ph["FILE_ID"].tolist()
    for st in STRATEGIES:
        os.makedirs(f"{out}/{st}", exist_ok=True)
        for at in ATLASES:
            urls = {i: f"{S3}/Outputs/cpac/{st}/rois_{at}/{i}_rois_{at}.1D" for i in ids}
            with ThreadPoolExecutor(48) as ex:
                raw = dict(zip(urls, ex.map(fetch, urls.values())))
            ts = {}
            for i, b in raw.items():
                if b:
                    a = np.loadtxt(io.BytesIO(b), comments="#", dtype=np.float32, ndmin=2)
                    ts[i] = a
            with open(f"{out}/{st}/{at}.pkl", "wb") as f:
                pickle.dump(ts, f, protocol=4)
            shapes = sorted({v.shape[1] for v in ts.values()})
            print(st, at, len(ts), "subjects, ROIs", shapes, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

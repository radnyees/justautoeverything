"""
projects/fmri_connectivity/data/build_adhd200.py -- ADHD-200 Preprocessed (Neuro Bureau, Athena
pipeline) ROI time courses for 6 atlases (training "filtfix" + test releases) + consolidated
phenotypics -> /data/fmri/adhd200/.

The archives are linked from NITRC (group 383, free login + an e-mail "Data Collection Agreement");
the files themselves are served from neurobureau.projects.nitrc.org.
    python projects/fmri_connectivity/data/build_adhd200.py /data/fmri/adhd200
Writes athena/<atlas>.pkl = {7-digit subject id: float32 (timepoints x ROIs)} using the band-pass
filtered run session_1_rest_1 (sfnwmrda*), and phenotypes.tsv.
"""
import io
import os
import pickle
import re
import sys
import tarfile
import urllib.request

import numpy as np

BASE = "https://neurobureau.projects.nitrc.org/Athena/"
PHENO = "https://www.nitrc.org/frs/download.php/9024/adhd200_preprocessed_phenotypics.tsv"
ATLASES = {"aal": "AAL", "cc200": "CC200", "cc400": "CC400", "ez": "EZ", "ho": "HO", "tt": "TT"}
PAT = re.compile(r"sfnwmrda(\d+)_session_1_rest_1_.*_TCs\.1D$")


def fetch(url):
    return urllib.request.urlopen(url, timeout=1800).read()


def parse(b):
    rows = [l.split() for l in b.decode().splitlines() if l.strip()]
    return np.asarray([[float(v) for v in r[2:]] for r in rows[1:]], dtype=np.float32)


def main(out):
    os.makedirs(f"{out}/athena", exist_ok=True)
    open(f"{out}/phenotypes.tsv", "wb").write(fetch(PHENO))
    for at, AT in ATLASES.items():
        ts = {}
        for fname, mode in [(f"ADHD200_{AT}_TCs_filtfix.tar.gz", "r:gz"), (f"ADHD200_{AT}_TCs_TestRelease.tar", "r:")]:
            tf = tarfile.open(fileobj=io.BytesIO(fetch(BASE + fname)), mode=mode)
            for m in tf.getmembers():
                g = PAT.search(m.name)
                if g and m.isfile():
                    ts[g.group(1).zfill(7)] = parse(tf.extractfile(m).read())
        with open(f"{out}/athena/{at}.pkl", "wb") as f:
            pickle.dump(ts, f, protocol=4)
        print(at, len(ts), "subjects; ROIs", sorted({v.shape[1] for v in ts.values()})[:5], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

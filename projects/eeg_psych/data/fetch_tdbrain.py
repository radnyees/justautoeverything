"""
projects/eeg_psych/data/fetch_tdbrain.py -- download the TDBRAIN V3.1 archive (van Dijk et al. 2022,
Sci Data; brainclinics.com/resources; CC BY 4.0 under the TDBRAIN Data Use Agreement) into the
private Modal volume. The zip is password-protected; the password is e-mailed to the registrant
after accepting the DUA. NEVER redistribute these files (DUA item 3) -- they stay in jae-data only.
    python projects/eeg_psych/data/fetch_tdbrain.py /data/eeg_psych/tdbrain
"""
import os
import shutil
import sys
import urllib.request

BASE = "https://brainclinics.com/images/downloads/"
FILES = ["TDBRAIN_participants_V3.1.xlsx", "TDBRAIN_Dataset_V3_1_Encr.zip"]


def main(out):
    os.makedirs(out, exist_ok=True)
    for f in FILES:
        dst = f"{out}/{f}"
        req = urllib.request.Request(BASE + f, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3600) as r, open(dst + ".part", "wb") as w:
            shutil.copyfileobj(r, w, length=16 << 20)
        os.replace(dst + ".part", dst)
        print(f, os.path.getsize(dst), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

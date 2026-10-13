"""
projects/eeg_psych/data/fetch_open.py -- download + unpack the OPEN resting-state EEG datasets into
/data/eeg_psych/raw/<dataset>/ and print their layout.

  aseeg  : ASEEG schizophrenia (Zenodo 18029536, CC BY 4.0) -- 51 SZ + 50 HC, 19 ch, 500 Hz, EO+EC
  msu    : Moscow State University adolescents (Borisov et al.) -- 45 SZ + 39 HC, 16 ch, 128 Hz, 1 min EC
  repod  : RepOD (Olejarczyk & Jernajczyk 2017, CC0) -- 14 SZ + 14 HC, 19 ch, 250 Hz, EC
  mumtaz : Mumtaz et al. 2016 MDD (figshare 4244171, CC BY 4.0) -- 34 MDD + 30 H, 19 ch, 256 Hz, EO+EC
    python projects/eeg_psych/data/fetch_open.py /data/eeg_psych/raw [datasets...]
"""
import collections
import json
import os
import ssl
import subprocess
import sys
import urllib.request


def get(url, dst, insecure=False):
    if os.path.exists(dst):
        return
    ctx = ssl._create_unverified_context() if insecure else None
    for t in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=1800, context=ctx) as r, open(dst + ".part", "wb") as w:
                while True:
                    b = r.read(16 << 20)
                    if not b:
                        break
                    w.write(b)
            os.replace(dst + ".part", dst)
            return
        except Exception as e:
            print("retry", url, e, flush=True)
    raise RuntimeError(url)


def fetch(name, out):
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    if name == "aseeg":
        for f in ["EO.rar", "EC.rar", "participants info.xlsx"]:
            get(f"https://zenodo.org/api/records/18029536/files/{f}/content", f"{d}/{f}")
        for f in ["EO.rar", "EC.rar"]:
            subprocess.run(["unar", "-q", "-f", "-o", d, f"{d}/{f}"], check=True)
    elif name == "msu":
        for f in ["norm.zip", "sch.zip"]:
            get(f"http://brain.bio.msu.ru/eeg_data/schizophrenia/{f}", f"{d}/{f}", insecure=True)
            subprocess.run(["unar", "-q", "-f", "-o", d, f"{d}/{f}"], check=True)
    elif name == "repod":
        meta = json.loads(urllib.request.urlopen(
            "https://repod.icm.edu.pl/api/datasets/:persistentId/?persistentId=doi:10.18150/repod.0107441",
            timeout=120).read())
        for f in meta["data"]["latestVersion"]["files"]:
            df = f["dataFile"]
            get(f"https://repod.icm.edu.pl/api/access/datafile/{df['id']}", f"{d}/{df['filename']}")
    elif name == "mumtaz":
        meta = json.loads(urllib.request.urlopen("https://api.figshare.com/v2/articles/4244171", timeout=120).read())
        for f in meta["files"]:
            get(f["download_url"], f"{d}/{f['name']}")
    # layout summary
    ext = collections.Counter()
    sample = []
    for root, _, files in os.walk(d):
        for f in files:
            ext[os.path.splitext(f)[1].lower()] += 1
            if len(sample) < 8 and not f.endswith((".rar", ".zip")):
                sample.append(os.path.relpath(os.path.join(root, f), d))
    print(name, dict(ext), sample, flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    for name in (sys.argv[2:] or ["aseeg", "msu", "repod", "mumtaz"]):
        fetch(name, out)

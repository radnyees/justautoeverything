"""
core/prep.py -- run a project's one-time data-building script on Modal (writes to jae-data).

    ./core/run.sh python core/prep.py projects/microbiome/data/build.py /data/microbiome [--size l]
"""
import argparse

import modal

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("args", nargs="*")
ap.add_argument("--size", default="m", choices=["m", "l"])
a = ap.parse_args()
res = modal.Function.from_name("jae", f"prep_{a.size}").remote(a.script, a.args)
print(res["log"])
raise SystemExit(res["returncode"])

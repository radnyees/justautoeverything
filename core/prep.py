"""
core/prep.py -- run a project's one-time data-building script on Modal (writes to jae-data).

    ./core/run.sh python core/prep.py projects/microbiome/data/build.py /data/microbiome [--size l]
"""
import argparse

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from bundle import bundle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("args", nargs="*")
ap.add_argument("--size", default="m", choices=["m", "l"])
a = ap.parse_args()
project = Path(a.script).parts[1] if a.script.startswith("projects/") else None
res = modal.Function.from_name("jae", f"pr_{a.size}").remote(
    a.script, a.args, bundle(project) if project else None)
print(res["log"])
raise SystemExit(res["returncode"])

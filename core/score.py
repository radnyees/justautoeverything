"""
core/score.py -- score one solution file on Modal (Dev CV, or the one-shot held-out Eval).

    ./core/run.sh python core/score.py microbiome crc_baxter seeds/default.py [--seed 1] [--mode dev|eval] [--report out.json]
Use --mode eval only for a frozen seed or a finished champion: Eval is scored once.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loop import project_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project")
ap.add_argument("task")
ap.add_argument("solution")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mode", default="dev", choices=["dev", "eval"])
ap.add_argument("--report", default=None, help="write the prediction report JSON here")
ap.add_argument("--time-limit", type=int, default=3600)
a = ap.parse_args()

import modal  # noqa: E402

size = project_cfg(a.project)["tasks"][a.task].get("size", "m")
sol = Path(a.solution)
if not sol.exists():
    sol = Path(__file__).parents[1] / "projects" / a.project / a.solution
t0 = time.time()
res = modal.Function.from_name("jae", f"evaluate_{size}").remote(
    a.project, a.task, sol.read_text(), a.mode, a.seed, bool(a.report), a.time_limit)
print(res["log"][-1500:])
print(f"{a.project}/{a.task} seed={a.seed} mode={a.mode} SCORE={res['score']} ({time.time() - t0:.0f}s)")
if a.report and res.get("report"):
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(res["report"])

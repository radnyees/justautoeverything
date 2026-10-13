"""core/check_task.py -- print a task's cohort summary from prepare.load (runs on Modal: /data).
    ./core/run.sh python core/prep.py core/check_task.py microbiome ibd_pooled"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from harness import EVAL_FRAC, grouped_stratified_split  # noqa: E402

project, task = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("prep", ROOT / "projects" / project / "prepare.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
ds = prep.load(task)
y, g = np.asarray(ds["y"]), np.asarray(ds["groups"]).astype(str)
sites = np.asarray(ds["sites"]).astype(str) if ds.get("sites") is not None else np.array(["-"] * len(y))
print(f"{task}: items {len(y)}, groups {len(set(g))}, cases {int(y.sum())}, controls {int((1 - y).sum())}")
for s in sorted(set(sites)):
    m = sites == s
    print(f"  {s}: items {m.sum()}, groups {len(set(g[m]))}, cases {int(y[m].sum())}, controls {int((1 - y[m]).sum())}")
gc = Counter(g)
print("  groups with >1 item:", sum(v > 1 for v in gc.values()), "max items per group:", max(gc.values()))
print("  example groups:", sorted(gc)[:3], [k for k, v in gc.items() if v > 1][:4])
if hasattr(ds["meta"], "columns") and "sample_location" in ds["meta"]:
    print("  sample_location:", ds["meta"]["sample_location"].value_counts().to_dict())
for seed in (1, 2, 3):
    dev, ev = grouped_stratified_split(ds["groups"], ds["strata"], EVAL_FRAC, seed)
    assert not set(g[dev]) & set(g[ev]), "group crosses Dev/Eval"
    print(f"  split {seed}: dev {len(dev)}, eval {len(ev)} (cases {int(y[ev].sum())}); no group crosses Dev/Eval")

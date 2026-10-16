"""core/probe_features.py -- EXPLORATORY (not pre-registered): which recording properties separate TDBRAIN
patients from controls? Computes the protocol-probe features (P2 set, EC and EO) for every subject of the
two TDBRAIN tasks and reports, per feature, the univariate AUROC and the group medians. Runs on Modal
(/data); writes only aggregates to /data/analysis/probe_features.json.
    ./core/run.sh python core/prep.py core/probe_features.py"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["length_s", "present", "line50_rel", "line60_rel", "dc_offset", "flat_frac", "clip_frac",
         "log_sd", "log_emg_55_100", "log_power_1_45"]


def main():
    spec = importlib.util.spec_from_file_location("prep", ROOT / "projects" / "eeg_psych" / "prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    ns = {"PARAMS": {"set": "P2"}}
    exec((ROOT / "projects" / "eeg_psych" / "probes" / "protocol_probe.py").read_text(), ns)
    out = {}
    for task in ("mdd_tdbrain", "adhd_tdbrain"):
        ds = prep.load(task)
        F = ns["featurize"](ds["items"], ds["ctx"])
        y = np.asarray(ds["y"]).astype(int)
        cols = [f"{c}:{n}" for c in ds["ctx"]["conditions"] for n in NAMES]
        res = []
        for j, name in enumerate(cols):
            v = F[:, j]
            if np.ptp(v) == 0:
                continue
            a = roc_auc_score(y, v)
            res.append({"feature": name, "auroc": float(a), "separation": float(abs(a - 0.5)),
                        "median_patients": float(np.median(v[y == 1])), "median_controls": float(np.median(v[y == 0]))})
        res.sort(key=lambda r: -r["separation"])
        out[task] = {"n": int(len(y)), "features": res}
        # absolute mains-noise power (the pre-registered P1 measures it RELATIVE to 1-45 Hz power, which
        # mixes mains noise with EEG power): absolute = relative x total, per condition
        for c in ds["ctx"]["conditions"]:
            rel50, tot = F[:, cols.index(f"{c}:line50_rel")], np.exp(F[:, cols.index(f"{c}:log_power_1_45")])
            v = np.log(rel50 * tot + 1e-12)
            a = roc_auc_score(y, v)
            res.append({"feature": f"{c}:log_line50_abs", "auroc": float(a), "separation": float(abs(a - 0.5)),
                        "median_patients": float(np.median(v[y == 1])), "median_controls": float(np.median(v[y == 0]))})
        res.sort(key=lambda r: -r["separation"])
        print(task, json.dumps([r for r in res if "line50" in r["feature"] or "power_1_45" in r["feature"]]), flush=True)
    (Path("/data") / "analysis").mkdir(exist_ok=True)
    (Path("/data") / "analysis" / "probe_features.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

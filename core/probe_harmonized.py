"""
core/probe_harmonized.py -- EXPLORATORY follow-up to the pre-registered TDBRAIN probe (not itself
pre-registered). Re-scores the frozen seed and loop champion of mdd_tdbrain / adhd_tdbrain on harmonized
recordings (same length, mains noise notched, DC removed -- projects/eeg_psych/probes/harmonize_wrapper.py),
and re-runs the P1 probe on the same harmonized data to check that harmonization removed the technical
signal. Splits 1-3, Eval scored once per variant. Appends a section to results/robustness/PROBE.md.
    ./core/run.sh python core/probe_harmonized.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from loop import evaluate  # noqa: E402
from probe import union_bootstrap  # noqa: E402

TASKS = ["mdd_tdbrain", "adhd_tdbrain"]
OUT = ROOT / "results" / "robustness" / "probe"
PD = ROOT / "projects" / "eeg_psych"


def main(variant="harmonized"):
    """variant: 'harmonized' (length, mains, DC) or 'harmscaled' (+ each recording scaled to unit SD)."""
    wrap = "\n" + ("HARMONIZE_SCALE = True\n" if variant == "harmscaled" else "") + \
        (PD / "probes" / "harmonize_wrapper.py").read_text()          # leading newline: code may not end in one
    probe = "PARAMS = {'set': 'P1'}\n" + (PD / "probes" / "protocol_probe.py").read_text()
    seed = (PD / "seeds" / "default.py").read_text()
    jobs = {}
    with ThreadPoolExecutor(6) as ex:
        for t in TASKS:
            for s in (1, 2, 3):
                champ = (PD / "runs" / t / f"s{s}" / "solution.py").read_text()
                for name, code in (("seed", seed), ("champion", champ), ("probeP1", probe)):
                    f = OUT / f"{variant}_{t}_{name}_s{s}.json"
                    if not f.exists():
                        jobs[f] = ex.submit(evaluate, "eeg_psych", t, "m", code + wrap, s, "eval", True, 1500)
        for f, fut in jobs.items():
            score, log, rep = fut.result()
            if rep is None:
                raise SystemExit(f"failed: {f.name}\n{log[-800:]}")
            f.write_text(rep if isinstance(rep, str) else json.dumps(rep))
            print(f"{f.stem}: {score:.4f}", flush=True)

    def rep(path):
        r = json.loads(Path(path).read_text())
        return {"y": np.asarray(r["y"]).astype(int), "p": np.asarray(r["predictions"], float),
                "g": [str(x) for x in r["groups"]]}

    title = {"harmonized": "harmonized recordings (length, mains noise, DC offset)",
             "harmscaled": "harmonized AND amplitude-normalized recordings (each scaled to unit median SD)"}[variant]
    L = ["", f"## Exploratory follow-up (not pre-registered): {title}", "",
         "Every recording cut to its first 100 s, mains noise notched (50/60 Hz and harmonics), per-channel",
         "DC offset removed" + (", then divided by its median channel SD" if variant == "harmscaled" else "") +
         ", before the frozen solutions see it (probes/harmonize_wrapper.py; solution code unchanged). The P1",
         "probe re-run on the same data shows how much technical signal remains.", "",
         "| task | model | Eval AUROC raw (mean of 3 splits) | harmonized | 95% CI harmonized |", "|---|---|---|---|---|"]
    for t in TASKS:
        rows = {}
        for name, raw_tpl in (("probeP1", OUT / "{t}_P1_s{s}.json"),
                              ("seed", PD / "runs" / "{t}" / "s{s}" / "final" / "seed.json"),
                              ("champion", PD / "runs" / "{t}" / "s{s}" / "final" / "champion.json")):
            raw = [rep(str(raw_tpl).format(t=t, s=s)) for s in (1, 2, 3)]
            har = [rep(OUT / f"{variant}_{t}_{name}_s{s}.json") for s in (1, 2, 3)]
            for a, b in zip(raw, har):
                assert a["g"] == b["g"], "raw and harmonized scored on different Eval items"
            m_raw = float(np.mean([roc_auc_score(r["y"], r["p"]) for r in raw]))
            m, lo, hi = union_bootstrap(har)
            rows[name] = (m_raw, m)
            L.append(f"| {t} | {name} | {m_raw:.3f} | {m:.3f} | {lo:.3f} .. {hi:.3f} |")
        L.append(f"| {t} | loop gain (champion - seed) | {rows['champion'][0] - rows['seed'][0]:+.3f} | "
                 f"{rows['champion'][1] - rows['seed'][1]:+.3f} | |")
    (ROOT / "results" / "robustness" / "PROBE.md").open("a").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "harmonized")

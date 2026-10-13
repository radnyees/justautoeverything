"""
core/shortcuts.py -- how far do NON-BIOLOGICAL shortcuts go? (runs on Modal: needs /data)

For the multi-site tasks, fits logistic regression on shortcut-only inputs -- site/study one-hot,
demographics (age, sex; ABIDE also head motion), and both -- on each split's dev pool and scores
the held-out Eval set (same splits as the loops) with POOLED AUROC (protocol v1) and WITHIN-SITE
AUROC (protocol v2). Also writes item -> site/study maps so archived v1 champion predictions can be
rescored within site (core/report_robustness.py).

    ./core/run.sh python core/prep.py core/shortcuts.py          # writes /data/analysis/shortcuts.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from harness import EVAL_FRAC, grouped_stratified_split  # noqa: E402
from metrics import within_site_auroc  # noqa: E402
import importlib.util  # noqa: E402

DATA = Path("/data")


def prepare(project):
    spec = importlib.util.spec_from_file_location("prep_" + project, ROOT / "projects" / project / "prepare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def demographics(task, items):
    if task == "adhd_adhd200":
        ph = pd.read_csv(DATA / "fmri" / "adhd200" / "phenotypes.tsv", sep="\t")
        ph["sid"] = ph["ScanDir ID"].astype(str).str.zfill(7)
        ph = ph.drop_duplicates("sid").set_index("sid").reindex(items)
        return pd.DataFrame({"age": pd.to_numeric(ph["Age"], errors="coerce"),
                             "sex": pd.to_numeric(ph["Gender"], errors="coerce")}, index=items)
    if task == "autism_abide":
        ph = pd.read_csv(DATA / "fmri" / "abide" / "phenotypes.csv")
        ph = ph.drop_duplicates("FILE_ID")
        ph.index = ph["FILE_ID"].astype(str)
        ph = ph.reindex(items)
        return pd.DataFrame({"age": pd.to_numeric(ph["AGE_AT_SCAN"], errors="coerce"),
                             "sex": pd.to_numeric(ph["SEX"], errors="coerce"),
                             "mean_fd": pd.to_numeric(ph["func_mean_fd"], errors="coerce")}, index=items)
    if task == "csf_ad":
        m = pd.read_parquet(DATA / "proteomics_ad" / "csf.meta.parquet")
        print("csf meta columns:", list(m.columns), flush=True)
        cols = {}
        for c in m.columns:
            lc = c.lower()
            if lc in ("age", "age_at_visit", "age_years"):
                cols["age"] = pd.to_numeric(m[c], errors="coerce")
            if lc in ("sex", "gender"):
                cols["sex"] = m[c].astype(str).str.upper().str[0].map({"M": 1, "F": 0, "1": 1, "2": 0})
        d = pd.DataFrame(cols, index=m.index)
        return d.reindex(items) if len(d.columns) else None
    return None                                   # ibd_pooled: no demographics in MicrobiomeHD


def design(kind, sites, demo):
    parts = []
    if kind in ("site", "site+demo"):
        parts.append(pd.get_dummies(pd.Series(sites).astype(str)).to_numpy(float))
    if kind in ("demo", "site+demo"):
        d = demo.to_numpy(float)
        d = np.where(np.isnan(d), np.nanmedian(d, axis=0), d)
        parts.append(d)
    return np.hstack(parts)


def main():
    out = {}
    for project, task in [("fmri_connectivity", "adhd_adhd200"), ("fmri_connectivity", "autism_abide"),
                          ("microbiome", "ibd_pooled"), ("proteomics_ad", "csf_ad")]:
        prep = prepare(project)
        ds = prep.load(task)
        items = list(ds["items"].index) if hasattr(ds["items"], "iloc") else list(ds["items"])
        items = [str(i) for i in items]
        y, sites = np.asarray(ds["y"]), np.asarray(ds["sites"]).astype(str)
        demo = demographics(task, items)
        kinds = ["site"] + (["demo", "site+demo"] if demo is not None else [])
        res = {"n": len(y), "kinds": kinds, "splits": {}, "site_of": dict(zip(items, sites.tolist())),
               "group_site": dict(zip([str(g) for g in ds["groups"]], sites.tolist())),
               "prevalence_by_site": {s: [float(y[sites == s].mean()), int((sites == s).sum())]
                                      for s in np.unique(sites)}}
        for seed in (1, 2, 3):
            dev, ev = grouped_stratified_split(ds["groups"], ds["strata"], EVAL_FRAC, seed)
            r = {}
            for k in kinds:
                X = design(k, sites, demo)
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[dev], y[dev])
                p = clf.predict_proba(X[ev])[:, 1]
                r[k] = {"pooled_auroc": float(roc_auc_score(y[ev], p)),
                        "within_site_auroc": float(within_site_auroc(y[ev], p, sites[ev]))}
            res["splits"][seed] = r
            print(task, seed, json.dumps(r), flush=True)
        out[task] = res
    (DATA / "analysis").mkdir(exist_ok=True)
    (DATA / "analysis" / "shortcuts.json").write_text(json.dumps(out))
    print("wrote /data/analysis/shortcuts.json", flush=True)


if __name__ == "__main__":
    main()

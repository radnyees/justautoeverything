"""
core/confounds.py -- do age/sex explain the Eval results? (local; aggregate output only)

Input: per-item demographics + labels from core/confounds_data.py (/tmp/confounds.json, downloaded
from the jae-data volume; not committed) and the saved Eval predictions of seed and champion.
For ADHD-200, ABIDE, TDBRAIN MDD and TDBRAIN ADHD, per split (same Eval sets as the loops):
  1. balance: age (mean) and % male per group in the task cohort;
  2. demographics-only model: logistic regression on age + sex (+ site one-hot) fitted on the
     split's dev pool, scored on Eval (within-site AUROC where the task is multi-site);
  3. do predictions track demographics? among Eval CONTROLS (labels not involved): within-site
     Spearman correlation of the prediction with age, and within-site AUROC of the prediction
     for sex (male = positive);
  4. demographic-adjusted AUROC: the prediction is residualized on age + sex (+ site dummies) by
     least squares over all Eval items (label-free), then scored like the task;
  5. likelihood-ratio test on Eval: y ~ age + sex (+ site) versus + standardized prediction.
ASEEG and the TUAB build carry no age/sex, so they cannot be checked here.
    python core/confounds.py [/tmp/confounds.json]
Writes results/robustness/CONFOUNDS.md, confounds.csv, figures/fig13_confounds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from metrics import within_site_auroc  # noqa: E402

RB = ROOT / "results" / "robustness"
TASKS = {"adhd_adhd200": "fmri_connectivity", "autism_abide": "fmri_connectivity",
         "mdd_tdbrain": "eeg_psych", "adhd_tdbrain": "eeg_psych"}


def _auc(y, p, sites):
    return within_site_auroc(y, p, sites) if sites is not None else float(roc_auc_score(y, p))


def _design(df, with_site):
    X = [df["age"].values, (df["sex"] == "M").astype(float).values]
    if with_site:
        X.append(pd.get_dummies(df["site"].astype(str), drop_first=True).to_numpy(float).T)
        return np.column_stack([X[0], X[1], X[2].T]) if X[2].size else np.column_stack(X[:2])
    return np.column_stack(X)


def _loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _within_site(v, sites):
    v = np.asarray(v, float).copy()
    if sites is None:
        return v - v.mean()
    for s in np.unique(sites):
        m = sites == s
        v[m] -= v[m].mean()
    return v


def analyse(path):
    demo = json.loads(Path(path).read_text())["demographics"]
    rows, balance = [], []
    for task, project in TASKS.items():
        D = pd.DataFrame.from_dict(demo[task], orient="index")
        D.index = D.index.astype(str)
        multisite = D["site"].notna().all()
        for g, gg in D.groupby("y"):
            balance.append({"task": task, "group": "case" if g == 1 else "control", "n": len(gg),
                            "age_mean": gg["age"].mean(), "age_sd": gg["age"].std(),
                            "pct_male": 100 * (gg["sex"] == "M").mean()})
        for s in (1, 2, 3):
            run = ROOT / "projects" / project / "runs" / task / f"s{s}" / "final"
            reps = {a: json.loads((run / f"{a}.json").read_text()) for a in ("seed", "champion")}
            ev = [str(x) for x in reps["seed"]["groups"]]
            dev = [i for i in D.index if i not in set(ev)]
            E, V = D.loc[ev], D.loc[dev]
            ok_e = E["age"].notna() & E["sex"].notna()
            ok_v = V["age"].notna() & V["sex"].notna()
            sites_e = E["site"].astype(str).values if multisite else None
            y_e = E["y"].values.astype(int)
            # 2. demographics-only model fitted on the dev pool
            Xv = _design(pd.concat([V[ok_v], E[ok_e]]), multisite)
            nv = int(ok_v.sum())
            clf = LogisticRegression(C=1.0, max_iter=10000)
            mu, sd = Xv[:nv].mean(0), Xv[:nv].std(0) + 1e-9
            clf.fit((Xv[:nv] - mu) / sd, V[ok_v]["y"].values.astype(int))
            pd_e = clf.predict_proba((Xv[nv:] - mu) / sd)[:, 1]
            demo_auc = _auc(y_e[ok_e.values], pd_e, None if sites_e is None else sites_e[ok_e.values])
            for arm, rep in reps.items():
                assert [str(x) for x in rep["groups"]] == ev
                pr = np.asarray(rep["predictions"], float)
                m = ok_e.values
                ye, pe, se_ = y_e[m], pr[m], (None if sites_e is None else sites_e[m])
                Ed = E[ok_e]
                ctrl = ye == 0
                # 3. predictions vs demographics among controls (within site)
                rho = st.spearmanr(_within_site(pe[ctrl], None if se_ is None else se_[ctrl]),
                                   _within_site(Ed["age"].values[ctrl], None if se_ is None else se_[ctrl])).statistic
                male = (Ed["sex"].values[ctrl] == "M").astype(int)
                sex_auc = (_auc(male, pe[ctrl], None if se_ is None else se_[ctrl])
                           if 0 < male.sum() < len(male) else np.nan)
                # 4. demographic-adjusted AUROC
                Xd = _design(Ed, multisite)
                A = np.column_stack([np.ones(len(pe)), Xd])
                beta, *_ = np.linalg.lstsq(A, pe, rcond=None)
                resid = pe - A @ beta
                # 5. likelihood-ratio test of the prediction given demographics (+ site)
                Xs = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9)
                z = (pe - pe.mean()) / (pe.std() + 1e-12)
                lr = 2 * (_loglik(np.column_stack([Xs, z]), ye) - _loglik(Xs, ye))
                rows.append({"task": task, "split": s, "arm": arm, "n_eval": int(m.sum()),
                             "auc_reported": rep["score"], "auc_complete_cases": _auc(ye, pe, se_),
                             "auc_demographics_only": demo_auc, "auc_adjusted": _auc(ye, resid, se_),
                             "rho_age_controls": rho, "auc_sex_controls": sex_auc,
                             "lr_chi2": lr, "lr_p": float(st.chi2.sf(max(lr, 0), 1))})
    return pd.DataFrame(rows), pd.DataFrame(balance)


def write(df, bal):
    RB.mkdir(parents=True, exist_ok=True)
    df.to_csv(RB / "confounds.csv", index=False, float_format="%.4f")
    L = ["# Do age and sex explain the Eval results?", "",
         "Tasks with per-subject age and sex: ADHD-200, ABIDE, TDBRAIN MDD, TDBRAIN ADHD (ASEEG and the",
         "TUAB build have no demographics, so they are not checked). Neither age nor sex is ever given",
         "to a solution, but brain signals encode both, so a model can still learn them. Method in",
         "`core/confounds.py`; within-site metrics for the multi-site tasks.", "",
         "## Cohort balance", "", "| task | group | n | age, mean (sd) | % male |", "|---|---|---|---|---|"]
    for r in bal.itertuples():
        L.append(f"| {r.task} | {r.group} | {r.n} | {r.age_mean:.1f} ({r.age_sd:.1f}) | {r.pct_male:.0f} |")
    L += ["", "## Eval, mean over 3 splits", "",
          "| task | arm | AUROC (reported) | age+sex only (fit on dev) | AUROC adjusted for age+sex | "
          "rho(prediction, age) in controls | AUROC(prediction -> male) in controls | LR test p (splits) |",
          "|---|---|---|---|---|---|---|---|"]
    for (t, a), g in df.groupby(["task", "arm"], sort=False):
        L.append(f"| {t} | {a} | {g.auc_reported.mean():.3f} | {g.auc_demographics_only.mean():.3f} | "
                 f"{g.auc_adjusted.mean():.3f} | {g.rho_age_controls.mean():+.2f} | {g.auc_sex_controls.mean():.3f} | "
                 + " / ".join(f"{p:.3g}" for p in g.lr_p) + " |")
    gain = df.pivot_table(index=["task", "split"], columns="arm", values=["auc_complete_cases", "auc_adjusted"])
    L += ["", "## Loop gain before and after adjustment (champion - seed, mean over splits)", "",
          "| task | gain (complete cases) | gain after age+sex adjustment |", "|---|---|---|"]
    for t in df["task"].unique():
        g = gain.loc[t]
        L.append(f"| {t} | {(g[('auc_complete_cases', 'champion')] - g[('auc_complete_cases', 'seed')]).mean():+.3f} | "
                 f"{(g[('auc_adjusted', 'champion')] - g[('auc_adjusted', 'seed')]).mean():+.3f} |")
    L += ["", "Reading guide: 'age+sex only' is what demographics alone achieve on the same Eval set;",
          "a prediction that tracks age or sex in controls (|rho| or |AUROC - 0.5| well above 0) carries",
          "demographic signal; 'adjusted' and the LR test ask whether the prediction still separates",
          "cases from controls once age and sex (and site) are accounted for."]
    (RB / "CONFOUNDS.md").write_text("\n".join(L) + "\n")


def figure(df):
    from report import GRID, INK, REF, plt, save
    tasks = list(df["task"].unique())
    fig, ax = plt.subplots(figsize=(7.4, 0.9 * len(tasks) + 1.4))
    cols = {"auc_demographics_only": ("age + sex only", "#9AA5B4", "s"),
            "auc_complete_cases": ("prediction", "#2563C9", "o"),
            "auc_adjusted": ("prediction adjusted for age + sex", "#D9860F", "D")}
    for i, t in enumerate(tasks[::-1]):
        for k, arm in enumerate(("seed", "champion")):
            g = df[(df.task == t) & (df.arm == arm)]
            y = i + (0.18 if arm == "champion" else -0.18)
            for c, (lab, colr, mk) in cols.items():
                if c == "auc_demographics_only" and arm == "champion":
                    continue
                ax.scatter([g[c].mean()], [y], s=46, marker=mk, color=colr if arm == "champion" else "white",
                           edgecolor=colr, linewidth=1.5, zorder=3,
                           label=(lab if (i == 0 and (arm == "champion" or c == "auc_demographics_only")) else None))
            ax.text(0.405, y, arm, va="center", fontsize=7.5, color=INK)
    ax.axvline(0.5, color=REF, lw=1, ls="--")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks[::-1], fontsize=8.5)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Eval AUROC, mean over 3 splits (within site for ADHD-200 / ABIDE); hollow = seed, filled = champion")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8)
    ax.set_title("Do age and sex explain the predictions?", loc="left", fontweight="bold", pad=26)
    save(fig, "fig13_confounds")


if __name__ == "__main__":
    df, bal = analyse(sys.argv[1] if len(sys.argv) > 1 else "/tmp/confounds.json")
    write(df, bal)
    figure(df)
    print(df.groupby(["task", "arm"], sort=False)[["auc_reported", "auc_complete_cases", "auc_demographics_only",
                                                  "auc_adjusted", "rho_age_controls", "auc_sex_controls",
                                                  "lr_p"]].mean().round(3).to_string())

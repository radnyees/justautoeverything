# Research brief (applies to every project)

You improve ONE file, `solution.py`, for a biomedical prediction task. A fixed harness
(`core/harness.py` + the project's `prepare.py`, which you cannot see or change) loads the data,
splits it, trains and scores your code. Your file must define:

```python
def featurize(items, ctx):          # OPTIONAL. Called once on all items the harness needs.
    ...                             # Must be label-free (labels are never passed in).
    return F                        # anything indexable per item: ndarray / DataFrame / list

def fit_predict(F_train, y_train, F_test, ctx):
    ...                             # fit on (F_train, y_train), predict F_test
    return predictions              # one per test item (see the project brief for the type)
```

`ctx` holds task metadata (sampling rate, channel names, atlas names, taxonomy, ...;
see the project brief) plus `meta_train` / `meta_test` (per-item non-label covariates such as
site or age, when the project provides them) and `groups_train` (subject ids of the training
items, for grouped inner CV). The harness runs grouped K-fold CV inside a dev pool and
scores the out-of-fold predictions. A held-out evaluation set (never shown to you) scores the
final champion once.

## Rules
- Never try to read labels, files or splits outside what the harness passes you; do not
  hard-code anything about specific subjects. Tune hyperparameters only with inner CV on the
  training fold (use `ctx["groups_train"]` for grouping), never against the score you are given.
- One focused, well-motivated change per round. Make it a real hypothesis about the data
  (physiology, biology, measurement), not random search.
- Be efficient: each candidate has a hard time limit (~25 min for the whole CV on the project's
  container). Cache expensive per-item computations inside `featurize`.
- Deterministic: fix random seeds.
- Available libraries: numpy, scipy, pandas, scikit-learn, statsmodels, lightgbm, xgboost,
  catboost, networkx, PyWavelets, antropy, mne, mne-connectivity, pyriemann, nilearn, torch (CPU),
  joblib. No network access, no installing packages.
- Print nothing large; the harness prints the score.

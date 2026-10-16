from core.generic_models import suggest_classifier


def suggest(trial, task):
    p = {"welch_sec": trial.suggest_categorical("welch_sec", [1, 2, 4]),     # seed 2 s
         "fmax": trial.suggest_categorical("fmax", [30, 45]),               # seed 45 Hz
         "conds": trial.suggest_categorical("conds", ["all", "first"])}     # seed: all conditions
    p.update(suggest_classifier(trial, "lr", 0.1))
    return p

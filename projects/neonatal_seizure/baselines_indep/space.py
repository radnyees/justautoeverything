from core.generic_models import suggest_classifier


def suggest(trial, task):
    p = {"win": trial.suggest_categorical("win", [4, 8, 16]),                     # seed 8 s
         "smooth": trial.suggest_categorical("smooth", [1, 3, 10, 30]),           # seed 10 s
         "ll": trial.suggest_categorical("ll", [True, False]),                    # seed feature groups
         "rms": trial.suggest_categorical("rms", [True, False]),
         "bp": trial.suggest_categorical("bp", [True, False]),
         "agg": trial.suggest_categorical("agg", ["max+median", "max", "median"])}  # seed max+median
    p.update(suggest_classifier(trial, "hgb", None))
    return p

from core.generic_models import suggest_classifier


def suggest(trial, task):
    p = {"center": trial.suggest_categorical("center", [True, False]),               # seed: True
         "min_present": trial.suggest_categorical("min_present", [0.5, 0.6, 0.7, 0.8, 0.9]),  # seed 0.7
         "fill_offset": trial.suggest_categorical("fill_offset", [0.5, 1.0, 2.0])}   # seed 1.0
    p.update(suggest_classifier(trial, "lr", 0.05))
    return p

from core.generic_models import suggest_classifier


def suggest(trial, task):
    p = {"level": trial.suggest_categorical("level", ["genus", "family", "otu"])}   # seed: genus
    p.update(suggest_classifier(trial, "rf", None))
    return p

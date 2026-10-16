from core.generic_models import suggest_classifier

ATLASES = ["aal", "cc200", "cc400", "ez", "ho", "tt"]        # options both fMRI tasks expose (seed: cc200)


def suggest(trial, task):
    p = {"atlas": trial.suggest_categorical("atlas", ATLASES),
         "strategy": trial.suggest_categorical("strategy", [0, 1])}    # index into ctx['strategies']
    p.update(suggest_classifier(trial, "lr", 0.001))
    return p

from baseline import SampleClassMixin
from optuna.trial import Trial
from tune_regressor import regressor_tuning_entities
from tune_classifier import classifier_tuning_entities
from typing import Iterable, Dict, Optional


def sample_models_with_params(
        trial: Trial,
        task: str, 
        all_except: Optional[Iterable] = None,
        include: Optional[Iterable] = None):
    
    valid_tasks: Iterable[str] = ["classification", "regression"]
    if task not in valid_tasks:
        raise ValueError(f"Invalid task {task}, expects tasks to be 'regression' or 'classification'")
    
    search_space = None
    if task == "regression":
        search_space: Dict[str, object] = dict(regressor_tuning_entities)

    else:
        search_space:  Dict[str, object] = dict(classifier_tuning_entities)

    tuner_obj: SampleClassMixin = trial.suggest_categorical("model_tuner", list(search_space.values()))
    model = tuner_obj.sample_model(trial)

    return model

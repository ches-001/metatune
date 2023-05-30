import inspect
from baseline import BaseTuner
from optuna.trial import Trial, FrozenTrial
from tune_regressor import regressor_tuning_entities, regressor_tuner_model_class_dict
from tune_classifier import classifier_tuning_entities, classifier_tuner_model_class_dict
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
        search_space: Dict[str, object] = regressor_tuning_entities

    else:
        search_space:  Dict[str, object] = classifier_tuning_entities

    tuner_obj: BaseTuner = trial.suggest_categorical("model_tuner", list(search_space.values()))
    model = tuner_obj.sample_model(trial)

    return model


def make_sampled_model(best_trial: FrozenTrial, **kwargs):
    model_tuner = best_trial.params["model_tuner"]
    tuner_model_class_dict = {**regressor_tuner_model_class_dict, **classifier_tuner_model_class_dict}
    model_class = tuner_model_class_dict[model_tuner.__class__.__name__]

    model_params_names = list(inspect.signature(model_class.__dict__["__init__"]).parameters.keys())
    best_params_dict = {
        k.replace(f"{model_tuner.__class__.__name__}_", "") : v 
        for k, v in best_trial.params.items() 
        if k.replace(f"{model_tuner.__class__.__name__}_", "") in model_params_names
        }

    return model_class(**best_params_dict)

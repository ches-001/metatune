import inspect
from baseline import BaseTuner, TrialCheckMixin
from optuna.trial import Trial, FrozenTrial
from tune_regressor import regressor_tuning_entities, regressor_tuner_model_class_dict
from tune_classifier import classifier_tuning_entities, classifier_tuner_model_class_dict
from typing import Iterable, Dict, Optional, Any, Callable


class Sample(TrialCheckMixin):

    def __init__(
            self, 
            task: str,
            custom_tuners: Optional[Iterable[Dict[BaseTuner, Any]]]=None, 
            excluded: Optional[Iterable[BaseTuner]]=None,
            custom_only: bool=False):
        
        valid_tasks: Iterable[str] = ["classification", "regression"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task {task}, expects tasks to be 'regression' or 'classification'")
        
        self.task = task
        self.custom_tuners = custom_tuners
        self.excluded = excluded
        self.custom_only = custom_only

        if self.task == "regression":
            self.search_space: Dict[str, object] = regressor_tuning_entities
            self.tuner_model_map: Dict[str, Callable] = regressor_tuner_model_class_dict

        else:
            self.search_space:  Dict[str, object] = classifier_tuning_entities
            self.tuner_model_map: Dict[str, Callable] = classifier_tuner_model_class_dict
            

    def check_tuner_data_compatibility(self, X: Iterable, y: Iterable) -> Iterable[BaseTuner]:
        pass


    def sample_models_with_params(self,trial: Trial) -> Any:
        super().in_trial(trial)
        tuner_obj: BaseTuner = trial.suggest_categorical("model_tuner", list(self.search_space.values()))
        model = tuner_obj.sample_model(trial)

        return model


    def build_sampled_model(self, best_trial: FrozenTrial, **kwargs) -> Any:
        model_tuner: BaseTuner = best_trial.params["model_tuner"]
        model_class = self.tuner_model_map[model_tuner.__class__.__name__]

        model_params_names = list(inspect.signature(model_class.__dict__["__init__"]).parameters.keys())
        best_params_dict = {
            k.replace(f"{model_tuner.__class__.__name__}_", "") : v 
            for k, v in best_trial.params.items() 
            if k.replace(f"{model_tuner.__class__.__name__}_", "") in model_params_names
            }

        params = {**kwargs, **best_params_dict}
        return model_class(**params)

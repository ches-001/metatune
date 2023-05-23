from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    BaggingClassifier, 
    VotingClassifier
)

@dataclass
class RandomForestClassifierModel(SampleClassMixin):
    n_estimators_space: Iterable[int] = (1, 200)
    criterion_space: Iterable[str] = ("gini", "entropy", "log_loss")
    max_depth_space: Iterable[int] = (2, 1000)
    min_samples_split_space: Iterable[Union[int, float]] = (1e-4, 1.0)
    min_samples_leaf_space: Iterable[Union[int, float]] = (1e-4, 1.0)
    min_weight_fraction_leaf_space: Iterable[float] = (0.0, 0.5)
    max_features_space: Iterable[str] = ("sqrt", "log2", None)
    max_leaf_nodes_space: Iterable[int] = (1, 1000)
    min_impurity_decrease_space: Iterable[float] = (0.0, 1.0)
    bootstrap_space: Iterable[bool] = (True, False)
    oob_score_space: Iterable[bool] = (True, False)
    class_weight_space: Iterable[str] = ("balanced", )
    ccp_alpha_space: Iterable[float] = (0.0, 1.0)
    max_samples_space: Iterable[Union[int, float]] = (0.1, 1.0)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)

        is_space_type: Callable = lambda space, type : all(list(map(lambda x: isinstance(x, type), space)))
        
        params = {}
        params["n_estimators"] = trial.suggest_int("n_estimators", *self.n_estimators_space, log=False)
        params["criterion"] = trial.suggest_categorical("criterion", self.criterion_space)
        params["max_depth"] = trial.suggest_int("max_depth", *self.max_depth_space, log=False)

        if is_space_type(self.min_samples_split_space, float):
            params["min_samples_split"] = trial.suggest_float("min_samples_split", *self.min_samples_split_space, log=False)
        else:
            params["min_samples_split"] = trial.suggest_int("min_samples_split", *self.min_samples_split_space, log=False)

        if is_space_type(self.min_samples_leaf_space, float):
            params["min_samples_leaf"] = trial.suggest_float("min_samples_leaf", *self.min_samples_leaf_space, log=False)
        else:
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", *self.min_samples_leaf_space, log=False)

        params["min_weight_fraction_leaf"] = trial.suggest_float("min_weight_fraction_leaf", *self.min_weight_fraction_leaf_space, log=False)
        params["max_features"] = trial.suggest_categorical("max_features", self.max_features_space)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", *self.max_leaf_nodes_space, log=False)
        params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", *self.min_impurity_decrease_space)
        params["bootstrap"] = trial.suggest_categorical("bootstrap", self.bootstrap_space)
        params["oob_score"] = trial.suggest_categorical("oob_score", self.oob_score_space)

        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", *self.ccp_alpha_space, log=False)

        if is_space_type(self.max_samples_space, float):
            params["max_samples"] = trial.suggest_float("max_samples", *self.max_samples_space, log=False)

        else:
            params["max_samples"] = trial.suggest_int("max_samples", *self.max_samples_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", RandomForestClassifier, params)
        self.model = model
        
        return model
    

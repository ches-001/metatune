from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DecisionTreeClassifierModel(SampleClassMixin):
    criterion_space: Iterable[str] = ("gini", "entropy", "log_loss")
    splitter_space: Iterable[str] = ("best", "random")
    max_depth_space: Iterable[int] = (2, 1000)
    min_samples_split_int_space: Iterable[int] = (2, 1000)
    min_samples_leaf_int_space: Iterable[int] = (1, 1000)
    min_samples_split_float_space: Iterable[float] = (0.0, 1.0)
    min_samples_leaf_float_space: Iterable[float] = (0.0, 1.0)
    min_weight_fraction_leaf_space: Iterable[float] = (0.0, 0.5)
    max_features_space: Iterable[str] = ("sqrt", "log2")
    max_leaf_nodes_space: Iterable[int] = (1, 1000)
    min_impurity_decrease_space: Iterable[float] = (0.0, 1.0)
    ccp_alpha_space: Iterable[float] = (0.0, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical("criterion", self.criterion_space)
        params["splitter"] = trial.suggest_categorical("splitter", self.splitter_space)
        params["max_depth"] = trial.suggest_int("max_depth", *self.max_depth_space, log=False)

        int_or_float = trial.suggest_categorical("int_or_float", ["int", "float"])
        if int_or_float == "int":
            params["min_samples_split"] = trial.suggest_int("min_samples_split_int", *self.min_samples_split_int_space, log=False)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf_int", *self.min_samples_leaf_int_space, log=False)
        else:
            params["min_samples_split"] = trial.suggest_float("min_samples_split_float", *self.min_samples_split_float_space, log=False)
            params["min_samples_leaf"] = trial.suggest_float("min_samples_leaf_float", *self.min_samples_leaf_float_space, log=False)
        
        params["min_weight_fraction_leaf"] = trial.suggest_float("min_weight_fraction_leaf", *self.min_weight_fraction_leaf_space, log=False)
        params["max_features"] = trial.suggest_categorical("max_features", self.max_features_space)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", *self.max_leaf_nodes_space, log=False)
        params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", *self.min_impurity_decrease_space, log=False)
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", *self.ccp_alpha_space, log=False)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = DecisionTreeClassifier(**params)
        
        self.model = model
        return model
    
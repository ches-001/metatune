from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from types import MappingProxyType
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


@dataclass
class DecisionTreeClassifierTuner(BaseTuner):
    criterion_space: Iterable[str] = ("gini", "entropy", "log_loss")
    splitter_space: Iterable[str] = ("best", "random")
    max_depth_space: Dict[str, Any] = MappingProxyType({"low":2, "high":1000, "step":1, "log":True})
    min_samples_split_space: Iterable[Union[int, float]] = MappingProxyType({"low":1e-4, "high":1.0, "step":None, "log":True})
    min_samples_leaf_space: Iterable[Union[int, float]] = MappingProxyType({"low":1e-4, "high":1.0, "step":None, "log":True})
    min_weight_fraction_leaf_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":0.5, "step":None, "log":False})
    max_features_space: Iterable[Optional[str]] = ("sqrt", "log2", None)
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    max_leaf_nodes_space: Dict[str, Any] = MappingProxyType({"low":2, "high":1000, "step":1, "log":True})
    min_impurity_decrease_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    ccp_alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    class_weight_space: Iterable[Optional[str]] = ("balanced", None)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
                
        params = {}
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        params["splitter"] = trial.suggest_categorical(f"{self.__class__.__name__}_splitter", self.splitter_space)
        params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", **dict(self.max_depth_space))

        if self.is_space_type(self.min_samples_split_space, float):
            params["min_samples_split"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_split", **dict(self.min_samples_split_space))
        else:
            params["min_samples_split"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_split", **dict(self.min_samples_split_space))

        if self.is_space_type(self.min_samples_leaf_space, float):
            params["min_samples_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_leaf", **dict(self.min_samples_leaf_space))
        else:
            params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", **dict(self.min_samples_leaf_space))

        params["min_weight_fraction_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_weight_fraction_leaf", **dict(self.min_weight_fraction_leaf_space))
        params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)

        if params["splitter"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
            

        params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", **dict(self.max_leaf_nodes_space))
        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", **dict(self.min_impurity_decrease_space))
        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", **dict(self.ccp_alpha_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", DecisionTreeClassifier, params)
        self.model = model
        return model
    

@dataclass
class ExtraTreeClassifierTuner(DecisionTreeClassifierTuner):
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(ExtraTreeClassifierTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(DecisionTreeClassifierTuner, self).sample_model(trial)

        params = self.sample_params(trial)
        model = super(DecisionTreeClassifierTuner, self).evaluate_sampled_model("classification", ExtraTreeClassifier, params)
        self.model = model
        return model
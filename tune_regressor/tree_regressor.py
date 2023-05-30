from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor


@dataclass
class DecisionTreeRegressorTuner(BaseTuner):
    criterion_space: Iterable[str] = ("squared_error", "friedman_mse", "absolute_error", "poisson")
    splitter_space: Iterable[str] = ("best", "random")
    max_depth_space: Iterable[int] = (2, 1000)
    min_samples_split_space: Iterable[Union[int, float]] = (1e-4, 1.0)
    min_samples_leaf_space: Iterable[Union[int, float]] = (1e-4, 1.0)
    min_weight_fraction_leaf_space: Iterable[float] = (0.0, 0.5)
    max_features_space: Iterable[Optional[str]] = ("sqrt", "log2", None)
    random_state_space: Iterable[int] = (0, 10000)
    max_leaf_nodes_space: Iterable[int] = (2, 1000)
    min_impurity_decrease_space: Iterable[float] = (0.0, 1.0)
    ccp_alpha_space: Iterable[float] = (0.0, 1.0)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        params["splitter"] = trial.suggest_categorical(f"{self.__class__.__name__}_splitter", self.splitter_space)
        params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", *self.max_depth_space, log=False)

        if self._is_space_type(self.min_samples_split_space, float):
            params["min_samples_split"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_split", *self.min_samples_split_space, log=False)
        else:
            params["min_samples_split"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_split", *self.min_samples_split_space, log=False)

        if self._is_space_type(self.min_samples_leaf_space, float):
            params["min_samples_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_leaf", *self.min_samples_leaf_space, log=False)
        else:
            params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", *self.min_samples_leaf_space, log=False)

        params["min_weight_fraction_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_weight_fraction_leaf", *self.min_weight_fraction_leaf_space, log=False)
        params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)
        if params["splitter"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", *self.max_leaf_nodes_space, log=False)
        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", *self.min_impurity_decrease_space, log=False)
        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", *self.ccp_alpha_space, log=False)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        
        params = self.sample_params(trial)
        model = super()._evaluate_sampled_model("regression", DecisionTreeRegressor, params)
        self.model = model
        return model
    

@dataclass
class ExtraTreeRegressorTuner(DecisionTreeRegressorTuner):
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(ExtraTreeRegressorTuner, self).sample_params(trial)
        
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(DecisionTreeRegressorTuner, self).sample_model(trial)
        
        params = self.sample_params(trial)
        model = super(DecisionTreeRegressorTuner, self)._evaluate_sampled_model("regression", ExtraTreeRegressor, params)
        self.model = model
        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    DecisionTreeRegressorTuner.__name__: DecisionTreeRegressor,
    ExtraTreeRegressorTuner.__name__: ExtraTreeRegressor,
}
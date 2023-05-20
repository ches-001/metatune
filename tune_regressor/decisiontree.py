from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.tree import DecisionTreeRegressor


@dataclass
class DecisionTreeRegressorModel(SampleClassMixin):
    criterion: Iterable[str] = ("squared_error", "friedman_mse", "absolute_error", "poisson")
    splitter: Iterable[str] = ("best", "random")
    max_depth: Iterable[int] = (2, 1000)
    min_samples_split: Iterable[int] = (2, 1000)
    min_samples_leaf: Iterable[int] = (1, 1000)
    min_samples_split_floats: Iterable[float] = (0.0, 1.0)
    min_samples_leaf_floats: Iterable[float] = (0.0, 1.0)
    min_weight_fraction_leaf: Iterable[float] = (0.0, 0.5)
    max_features: Iterable[str] = ("sqrt", "log2")
    max_leaf_nodes: Iterable[int] = (1, 1000)
    min_impurity_decrease: Iterable[float] = (0.0, 1.0)
    ccp_alpha: Iterable[float] = (0.0, 1.0)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical("criterion", self.criterion)
        params["splitter"] = trial.suggest_categorical("splitter", self.splitter)
        params["max_depth"] = trial.suggest_int("max_depth", *self.max_depth, log=False)

        int_or_float = trial.suggest_categorical("int_or_float", ["int", "float"])
        if int_or_float == "int":
            params["min_samples_split"] = trial.suggest_int("min_samples_split", *self.min_samples_split, log=False)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", *self.min_samples_leaf, log=False)
        else:
            params["min_samples_split"] = trial.suggest_float("min_samples_split_floats", *self.min_samples_split_floats, log=False)
            params["min_samples_leaf"] = trial.suggest_float("min_samples_leaf_floats", *self.min_samples_leaf_floats, log=False)
        
        params["min_weight_fraction_leaf"] = trial.suggest_float("min_weight_fraction_leaf", *self.min_weight_fraction_leaf, log=False)
        params["max_features"] = trial.suggest_categorical("max_features", self.max_features)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", *self.max_leaf_nodes, log=False)
        params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", *self.min_impurity_decrease, log=False)
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", *self.ccp_alpha, log=False)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = DecisionTreeRegressor(
            **params,)
        
        self.model = model
        return model

from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from types import MappingProxyType
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    BaggingRegressor, 
    HistGradientBoostingRegressor,
)
from ..tune_classifier import BaggingClassifierTuner

@dataclass
class RandomForestRegressorTuner(BaseTuner):
    n_estimators_space: Dict[str, Any] = MappingProxyType({"low":1, "high":200, "step":1, "log":True})
    criterion_space: Iterable[str] = ("squared_error", "absolute_error", "friedman_mse", "poisson")
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Dict[str, Any] = MappingProxyType({"low":10, "high":2000, "step":1, "log":True})
    min_samples_split_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    min_samples_leaf_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    min_weight_fraction_leaf_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":0.5, "step":None, "log":False})
    max_features_space: Iterable[str] = ("sqrt", "log2", None)
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Dict[str, Any] = MappingProxyType({"low":2, "high":10000, "step":1, "log":True})
    min_impurity_decrease_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    bootstrap_space: Iterable[bool] = (True, False)
    oob_score_space: Iterable[bool] = (True, False)
    set_random_state_space: Iterable[bool] = (False, )
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    ccp_alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    set_max_samples_space: Iterable[bool] = (True, False)
    max_samples_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", **dict(self.n_estimators_space))
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
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
        
        if self.is_valid_categorical_space(self.max_features_space):
            params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)
        else:
            if self.is_valid_float_space(self.max_features_space):
                params["max_features"] = trial.suggest_float(f"{self.__class__.__name__}_max_features", **dict(self.max_features_space))
            else:
                params["max_features"] = trial.suggest_int(f"{self.__class__.__name__}_max_features", **dict(self.max_features_space))

        set_max_leaf_node = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_node:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", **dict(self.max_leaf_nodes_space))

        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", **dict(self.min_impurity_decrease_space))
        params["bootstrap"] = trial.suggest_categorical(f"{self.__class__.__name__}_bootstrap", self.bootstrap_space)
        params["oob_score"] = trial.suggest_categorical(f"{self.__class__.__name__}_oob_score", self.oob_score_space)

        set_random_state = trial.suggest_categorical(f"{self.__class__.__name__}_set_random_state", self.set_random_state_space)
        if set_random_state:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", **dict(self.ccp_alpha_space))

        set_max_samples = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_samples", self.set_max_samples_space)
        if set_max_samples:
            if self.is_space_type(self.max_samples_space, float):
                params["max_samples"] = trial.suggest_float(f"{self.__class__.__name__}_max_samples", **dict(self.max_samples_space))

            else:
                params["max_samples"] = trial.suggest_int(f"{self.__class__.__name__}_max_samples", **dict(self.max_samples_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", RandomForestRegressor, params)
        self.model = model
        
        return model
    

@dataclass
class ExtraTreesRegressorTuner(RandomForestRegressorTuner):
     
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(ExtraTreesRegressorTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(RandomForestRegressorTuner, self).sample_model(trial)
        params = self.sample_params(trial)
        model = super(RandomForestRegressorTuner, self).evaluate_sampled_model("regression", ExtraTreesRegressor, params)
        self.model = model
        
        return model
    

@dataclass
class AdaBoostRegressorTuner(BaseTuner):
    estimator_space: Iterable[Optional[object]] = (None, )
    n_estimators_space: Dict[str, Any] = MappingProxyType({"low":1, "high":200, "step":1, "log":True})
    learning_rate_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1, "step":None, "log":True})
    loss_space: Iterable[str] = ("linear", "square", "exponential")
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["estimator"] = trial.suggest_categorical(f"{self.__class__.__name__}_estimator", self.estimator_space)
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", **dict(self.n_estimators_space))
        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", **dict(self.learning_rate_space))
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", AdaBoostRegressor, params)
        self.model = model

        return model
    

@dataclass
class GradientBoostingRegressorTuner(BaseTuner):
    loss_space: Iterable[str] = ("squared_error", "absolute_error", "huber", "quantile")
    learning_rate_space: Dict[str, Any] = MappingProxyType({"low":0.001, "high":1.0, "step":None, "log":True})
    n_estimators_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    subsample_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    criterion_space: Iterable[str] = ("friedman_mse", "squared_error")
    min_samples_split_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    min_samples_leaf_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    min_weight_fraction_leaf_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":0.5, "step":None, "log":False})
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Dict[str, Any] = MappingProxyType({"low":10, "high":2000, "step":1, "log":True})
    min_impurity_decrease_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    init_space: Iterable[Optional[object]] = (None, )
    max_features_space: Iterable[str] = ("sqrt", "log2")
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1, "step":None, "log":True})
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Iterable[Optional[int]] = MappingProxyType({"low":2, "high":10000, "step":1, "log":True})
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    set_n_iter_no_change_space: Iterable[bool] = (True, False)
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    ccp_alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", **dict(self.learning_rate_space))
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", **dict(self.n_estimators_space))
        params["subsample"] = trial.suggest_float(f"{self.__class__.__name__}_subsample", **dict(self.subsample_space))
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        if self.is_space_type(self.min_samples_split_space, float):
            params["min_samples_split"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_split", **dict(self.min_samples_split_space))
        else:
            params["min_samples_split"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_split", **dict(self.min_samples_split_space))

        if self.is_space_type(self.min_samples_leaf_space, float):
            params["min_samples_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_leaf", **dict(self.min_samples_leaf_space))
        else:
            params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", **dict(self.min_samples_leaf_space))

        params["min_weight_fraction_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_weight_fraction_leaf", **dict(self.min_weight_fraction_leaf_space))

        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
            params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", **dict(self.max_depth_space))

        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", **dict(self.min_impurity_decrease_space))
        params["init"] = trial.suggest_categorical(f"{self.__class__.__name__}_init", self.init_space)
        
        if self.is_valid_categorical_space(self.max_features_space):
            params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)
        else:
            if self.is_valid_float_space(self.max_features_space):
                params["max_features"] = trial.suggest_float(f"{self.__class__.__name__}_max_features", **dict(self.max_features_space))
            else:
                params["max_features"] = trial.suggest_int(f"{self.__class__.__name__}_max_features", **dict(self.max_features_space))

        set_max_leaf_nodes = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", **dict(self.max_leaf_nodes_space))

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        
        set_n_iter_no_change = trial.suggest_categorical(f"{self.__class__.__name__}_set_n_iter_no_change", self.set_n_iter_no_change_space)
        if set_n_iter_no_change:
            params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", **dict(self.ccp_alpha_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", GradientBoostingRegressor, params)
        self.model = model

        return model


@dataclass
class BaggingRegressorTuner(BaggingClassifierTuner):
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(BaggingRegressorTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(BaggingClassifierTuner, self).sample_model(trial)
        params = self.sample_params(trial)
        model = super(BaggingClassifierTuner, self).evaluate_sampled_model("regression", BaggingRegressor, params)
        self.model = model

        return model
    

@dataclass
class HistGradientBoostingRegressorTuner(BaseTuner):
    loss_space: Iterable[str] = ("squared_error", "absolute_error", "poisson", "quantile")
    quantile_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    learning_rate_space: Dict[str, Any] = MappingProxyType({"low":0.001, "high":1.0, "step":None, "log":True})
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":10, "high":1000, "step":1, "log":True})
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Iterable[Optional[int]] = MappingProxyType({"low":2, "high":10000, "step":1, "log":True})
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Dict[str, Any] = MappingProxyType({"low":10, "high":2000, "step":1, "log":True})
    min_samples_leaf_space: Dict[str, Any] = MappingProxyType({"low":1, "high":200, "step":1, "log":True})
    l2_regularization_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    max_bins_space: Dict[str, Any] = MappingProxyType({"low":10, "high":255, "step":1, "log":True})
    categorical_features_space: Iterable[Any] = (None, )
    monotonic_cst_space: Iterable[Any] = (None, )
    interaction_cst_space: Iterable[Any] = (None, )
    early_stopping_space: Iterable[bool] = ("auto", True, False)
    scoring_space: Iterable[Optional[Union[str, Callable]]] = ("loss", None)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)

        if params["loss"] == "quantile":
            params["quantile"] = trial.suggest_float(f"{self.__class__.__name__}_quantile", **dict(self.quantile_space))

        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", **dict(self.learning_rate_space))
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        
        set_max_leaf_nodes = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", **dict(self.max_leaf_nodes_space))
        
        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
            params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", **dict(self.max_depth_space))
        
        params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", **dict(self.min_samples_leaf_space))
        params["l2_regularization"] = trial.suggest_float(f"{self.__class__.__name__}_l2_regularization", **dict(self.l2_regularization_space))
        params["max_bins"] = trial.suggest_int(f"{self.__class__.__name__}_max_bins", **dict(self.max_bins_space))
        params["categorical_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_categorical_features", self.categorical_features_space)
        params["monotonic_cst"] = trial.suggest_categorical(f"{self.__class__.__name__}_monotonic_cst", self.monotonic_cst_space)
        params["interaction_cst"] = trial.suggest_categorical(f"{self.__class__.__name__}_interaction_cst", self.interaction_cst_space)
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["scoring"] = trial.suggest_categorical(f"{self.__class__.__name__}_scoring", self.scoring_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", HistGradientBoostingRegressor, params)
        self.model = model

        return model

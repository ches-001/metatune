from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    BaggingRegressor, 
    HistGradientBoostingRegressor,
)
from tune_classifier import BaggingClassifierTuner

@dataclass
class RandomForestRegressorTuner(BaseTuner):
    n_estimators_space: Iterable[int] = (1, 200)
    criterion_space: Iterable[str] = ("squared_error", "absolute_error", "friedman_mse", "poisson")
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Iterable[int] = (10, 2000)
    min_samples_split_space: Iterable[Union[int, float]] = (0.1, 1.0)
    min_samples_leaf_space: Iterable[Union[int, float]] = (0.1, 1.0)
    min_weight_fraction_leaf_space: Iterable[float] = (0.0, 0.5)
    max_features_space: Iterable[str] = ("sqrt", "log2", None)
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Iterable[int] = (1, 1000)
    min_impurity_decrease_space: Iterable[float] = (0.0, 1.0)
    bootstrap_space: Iterable[bool] = (True, False)
    oob_score_space: Iterable[bool] = (True, False)
    set_random_state_space: Iterable[bool] = (False, )
    random_state_space: Iterable[int] = (0, 10000)
    ccp_alpha_space: Iterable[float] = (0.0, 1.0)
    set_max_samples_space: Iterable[bool] = (True, False)
    max_samples_space: Iterable[Union[int, float]] = (0.1, 1.0)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", *self.n_estimators_space, log=False)
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
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
        
        if self.is_valid_categorical_space(self.max_features_space):
            params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)
        else:
            if self.is_valid_float_space(self.max_features_space):
                params["max_features"] = trial.suggest_float(f"{self.__class__.__name__}_max_features", *self.max_features_space, log=False)
            else:
                params["max_features"] = trial.suggest_int(f"{self.__class__.__name__}_max_features", *self.max_features_space, log=False)

        set_max_leaf_node = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_node:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", *self.max_leaf_nodes_space, log=False)

        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", *self.min_impurity_decrease_space)
        params["bootstrap"] = trial.suggest_categorical(f"{self.__class__.__name__}_bootstrap", self.bootstrap_space)
        params["oob_score"] = trial.suggest_categorical(f"{self.__class__.__name__}_oob_score", self.oob_score_space)

        set_random_state = trial.suggest_categorical(f"{self.__class__.__name__}_set_random_state", self.set_random_state_space)
        if set_random_state:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", *self.ccp_alpha_space, log=False)

        set_max_samples = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_samples", self.set_max_samples_space)
        if set_max_samples:
            if self._is_space_type(self.max_samples_space, float):
                params["max_samples"] = trial.suggest_float(f"{self.__class__.__name__}_max_samples", *self.max_samples_space, log=False)

            else:
                params["max_samples"] = trial.suggest_int(f"{self.__class__.__name__}_max_samples", *self.max_samples_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super()._evaluate_sampled_model("regression", RandomForestRegressor, params)
        self.model = model
        
        return model
    

@dataclass
class ExtraTreesRegressorTuner(RandomForestRegressorTuner):
     
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(ExtraTreesRegressorTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(RandomForestRegressorTuner, self).sample_model(trial)
        params = self.sample_params(trial)
        model = super(RandomForestRegressorTuner, self)._evaluate_sampled_model("regression", ExtraTreesRegressor, params)
        self.model = model
        
        return model
    

@dataclass
class AdaBoostRegressorTuner(BaseTuner):
    estimator_space: Iterable[Optional[object]] = (None, )
    n_estimators_space: Iterable[int] = (1, 200)
    learning_rate_space: Iterable[float] = (0.01, 1.0)
    loss_space: Iterable[str] = ("linear", "square", "exponential")
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["estimator"] = trial.suggest_categorical(f"{self.__class__.__name__}_estimator", self.estimator_space)
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", *self.n_estimators_space, log=False)
        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", *self.learning_rate_space, log=False)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super()._evaluate_sampled_model("regression", AdaBoostRegressor, params)
        self.model = model

        return model
    

@dataclass
class GradientBoostingRegressorTuner(BaseTuner):
    loss_space: Iterable[str] = ("squared_error", "absolute_error", "huber", "quantile")
    learning_rate_space: Iterable[float] = (0.001, 1.0)
    n_estimators_space: Iterable[int] = (1, 100)
    subsample_space: Iterable[float] = (0.1, 1.0)
    criterion_space: Iterable[str] = ("friedman_mse", "squared_error")
    min_samples_split_space: Iterable[Union[int, float]] = (0.1, 1.0)
    min_samples_leaf_space: Iterable[Union[int, float]] = (0.1, 1.0)
    min_weight_fraction_leaf_space: Iterable[float] = (0.0, 0.5)
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Iterable[int] = (10, 2000)
    min_impurity_decrease_space: Iterable[float] = (0.0, 1.0)
    init_space: Iterable[Optional[object]] = (None, )
    max_features_space: Iterable[str] = ("sqrt", "log2")
    alpha_space: Iterable[float] = (0.01, 1.0)
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Iterable[Optional[int]] = (1, 1000)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    set_n_iter_no_change_space: Iterable[bool] = (True, False)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    random_state_space: Iterable[int] = (0, 10000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    ccp_alpha_space: Iterable[float] = (0.0, 1.0)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", *self.learning_rate_space, log=False)
        params["n_estimators"] = trial.suggest_int(f"{self.__class__.__name__}_n_estimators", *self.n_estimators_space, log=False)
        params["subsample"] = trial.suggest_float(f"{self.__class__.__name__}_subsample", *self.subsample_space, log=False)
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_space)
        if self._is_space_type(self.min_samples_split_space, float):
            params["min_samples_split"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_split", *self.min_samples_split_space, log=False)
        else:
            params["min_samples_split"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_split", *self.min_samples_split_space, log=False)

        if self._is_space_type(self.min_samples_leaf_space, float):
            params["min_samples_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples_leaf", *self.min_samples_leaf_space, log=False)
        else:
            params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", *self.min_samples_leaf_space, log=False)

        params["min_weight_fraction_leaf"] = trial.suggest_float(f"{self.__class__.__name__}_min_weight_fraction_leaf", *self.min_weight_fraction_leaf_space, log=False)

        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
            params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", *self.max_depth_space, log=False)

        params["min_impurity_decrease"] = trial.suggest_float(f"{self.__class__.__name__}_min_impurity_decrease", *self.min_impurity_decrease_space)
        params["init"] = trial.suggest_categorical(f"{self.__class__.__name__}_init", self.init_space)
        
        if self.is_valid_categorical_space(self.max_features_space):
            params["max_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_max_features", self.max_features_space)
        else:
            if self.is_valid_float_space(self.max_features_space):
                params["max_features"] = trial.suggest_float(f"{self.__class__.__name__}_max_features", *self.max_features_space, log=False)
            else:
                params["max_features"] = trial.suggest_int(f"{self.__class__.__name__}_max_features", *self.max_features_space, log=False)

        set_max_leaf_nodes = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", *self.max_leaf_nodes_space, log=False)

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", *self.max_leaf_nodes_space, log=False)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", *self.validation_fraction_space, log=False)
        
        set_n_iter_no_change = trial.suggest_categorical(f"{self.__class__.__name__}_set_n_iter_no_change", self.set_n_iter_no_change_space)
        if set_n_iter_no_change:
            params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", *self.n_iter_no_change_space, log=False)
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["ccp_alpha"] = trial.suggest_float(f"{self.__class__.__name__}_ccp_alpha", *self.ccp_alpha_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super()._evaluate_sampled_model("regression", GradientBoostingRegressor, params)
        self.model = model

        return model


@dataclass
class BaggingRegressorTuner(BaggingClassifierTuner):
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(BaggingRegressorTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(BaggingClassifierTuner, self).sample_model(trial)
        params = self.sample_params(trial)
        model = super(BaggingClassifierTuner, self)._evaluate_sampled_model("regression", BaggingRegressor, params)
        self.model = model

        return model
    

@dataclass
class HistGradientBoostingRegressorTuner(BaseTuner):
    loss_space: Iterable[str] = ("squared_error", "absolute_error", "poisson", "quantile")
    quantile_space: Iterable[float] = (0.0, 1.0)
    learning_rate_space: Iterable[float] = (0.001, 1.0)
    max_iter_space: Iterable[int] = (10, 1000)
    set_max_leaf_nodes_space: Iterable[bool] = (True, False)
    max_leaf_nodes_space: Iterable[Optional[int]] = (1, 1000)
    set_max_depth_space: Iterable[bool] = (True, False)
    max_depth_space: Iterable[int] = (10, 2000)
    min_samples_leaf_space: Iterable[int] = (1, 200)
    l2_regularization_space: Iterable[float] = (0.0, 1.0)
    max_bins_space: Iterable[int] = (10, 255)
    categorical_features_space: Iterable[Any] = (None, )
    monotonic_cst_space: Iterable[Any] = (None, )
    interaction_cst_space: Iterable[Any] = (None, )
    early_stopping_space: Iterable[bool] = ("auto", True, False)
    scoring_space: Iterable[Optional[Union[str, Callable]]] = ("loss", None)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)

        if params["loss"] == "quantile":
            params["quantile"] = trial.suggest_float(f"{self.__class__.__name__}_quantile", *self.quantile_space, log=False)

        params["learning_rate"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate", *self.learning_rate_space, log=False)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        
        set_max_leaf_nodes = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_leaf_nodes", self.set_max_leaf_nodes_space)
        if set_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int(f"{self.__class__.__name__}_max_leaf_nodes", *self.max_leaf_nodes_space, log=False)
        
        set_max_depth = trial.suggest_categorical(f"{self.__class__.__name__}_set_max_depth", self.set_max_depth_space)
        if set_max_depth:
            params["max_depth"] = trial.suggest_int(f"{self.__class__.__name__}_max_depth", *self.max_depth_space, log=False)
        
        params["min_samples_leaf"] = trial.suggest_int(f"{self.__class__.__name__}_min_samples_leaf", *self.min_samples_leaf_space, log=False)
        params["l2_regularization"] = trial.suggest_float(f"{self.__class__.__name__}_l2_regularization", *self.l2_regularization_space, log=False)
        params["max_bins"] = trial.suggest_int(f"{self.__class__.__name__}_max_bins", *self.max_bins_space, log=False)
        params["categorical_features"] = trial.suggest_categorical(f"{self.__class__.__name__}_categorical_features", self.categorical_features_space)
        params["monotonic_cst"] = trial.suggest_categorical(f"{self.__class__.__name__}_monotonic_cst", self.monotonic_cst_space)
        params["interaction_cst"] = trial.suggest_categorical(f"{self.__class__.__name__}_interaction_cst", self.interaction_cst_space)
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["scoring"] = trial.suggest_categorical(f"{self.__class__.__name__}_scoring", self.scoring_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super()._evaluate_sampled_model("regression", HistGradientBoostingRegressor, params)
        self.model = model

        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    RandomForestRegressorTuner.__name__: RandomForestRegressor,
    ExtraTreesRegressorTuner.__name__: ExtraTreesRegressor,
    AdaBoostRegressorTuner.__name__: AdaBoostRegressor,
    GradientBoostingRegressorTuner.__name__: GradientBoostingRegressor,
    BaggingRegressorTuner.__name__: BaggingRegressor,
    HistGradientBoostingRegressorTuner.__name__: HistGradientBoostingRegressor,
}

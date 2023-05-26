from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, Union, Callable
from sklearn.linear_model import (
    LinearRegression,
    Lasso, 
    Ridge, 
    ElasticNet, 
    MultiTaskLasso, 
    MultiTaskElasticNet)

@dataclass
class LinearRegressionTuner(SampleClassMixin):
    fit_intercept_space: Iterable[bool] = (True, False)
    positive_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", LinearRegression, params)
        self.model = model

        return model
    

@dataclass
class LassoTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)
        params["selection"] = trial.suggest_categorical("selection", self.selection_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", Lasso, params)
        self.model = model

        return model

 
@dataclass
class RidgeTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    solver_space: Iterable[str] = ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs")
    positive_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)
        params["solver"] = trial.suggest_categorical("solver", self.solver_space)
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", Ridge, params)
        self.model = model

        return model
    

@dataclass
class ElasticNetTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[Union[bool, Iterable]] = (True, False, )
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
        params["precompute"] = trial.suggest_categorical("precompute", self.precompute_space)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)
        params["selection"] = trial.suggest_categorical("selection", self.selection_space)
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", ElasticNet, params)
        self.model = model

        return model
    

@dataclass
class MultiTaskLassoTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    selection_space: Iterable[str] = ("cyclic", "random")
    is_multitask: str = field(init=False, default=True)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["selection"] = trial.suggest_categorical("selection", self.selection_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", MultiTaskLasso, params, is_mulktitask=self.is_multitask)
        self.model = model

        return model
    

@dataclass
class MultiTaskElasticNetTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    selection_space: Iterable[str] = ("cyclic", "random")
    is_multitask: str = field(init=False, default=True)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["selection"] = trial.suggest_categorical("selection", self.selection_space)
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model(
            "regression", MultiTaskElasticNet, params, is_mulktitask=self.is_multitask)
        self.model = model

        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    LinearRegressionTuner.__name__: LinearRegression,
    LassoTuner.__name__: Lasso,
    RidgeTuner.__name__: Ridge,
    ElasticNetTuner.__name__: ElasticNet,
    MultiTaskLassoTuner.__name__: MultiTaskLasso,
    MultiTaskElasticNetTuner.__name__: MultiTaskElasticNet,
}
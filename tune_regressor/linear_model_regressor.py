import numpy as np
from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, Union, Callable
from types import MappingProxyType
from sklearn.linear_model import (
    LinearRegression,
    Lasso, 
    Ridge, 
    ElasticNet, 
    MultiTaskLasso, 
    MultiTaskElasticNet, 
    Lars,
    LassoLars,
    LassoLarsIC,
    HuberRegressor, 
    TheilSenRegressor,
    BayesianRidge,
    OrthogonalMatchingPursuit,
    ARDRegression,
    PassiveAggressiveRegressor,
    QuantileRegressor,
    SGDRegressor, 
    RANSACRegressor, 
    PoissonRegressor, 
    GammaRegressor, 
    TweedieRegressor)
from sklearn.base import RegressorMixin, BaseEstimator 


@dataclass
class LinearRegressionTuner(BaseTuner):
    fit_intercept_space: Iterable[bool] = (True, False)
    positive_space: Iterable[bool] = (True, False)
    
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", LinearRegression, params)
        self.model = model

        return model
    

@dataclass
class LassoTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)

        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Lasso, params)
        self.model = model

        return model

 
@dataclass
class RidgeTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    solver_space: Iterable[str] = ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs")
    positive_space: Iterable[bool] = (True, False)
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)

        if params["solver"] in ["sag", "saga"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))  
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Ridge, params)
        self.model = model

        return model
    

@dataclass
class ElasticNetTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[Union[bool, Iterable]] = (True, False, )
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", ElasticNet, params)
        self.model = model

        return model
    

@dataclass
class MultiTaskLassoTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    is_multitask: str = field(init=False, default=True)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", MultiTaskLasso, params, is_multitask=self.is_multitask)
        self.model = model

        return model
    

@dataclass
class MultiTaskElasticNetTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    is_multitask: str = field(init=False, default=True)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model(
            "regression", MultiTaskElasticNet, params, is_multitask=self.is_multitask)
        self.model = model

        return model
    

@dataclass
class LarsTuner(BaseTuner):
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    n_nonzero_coefs_space: Dict[str, Any] = MappingProxyType({"low":1, "high":500, "step":1, "log":True})
    eps_space: Dict[str, Any] = MappingProxyType({"low":np.finfo(float).eps, "high":1e-10, "step":None, "log":True})
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Dict[str, Any] = MappingProxyType({"low":1e-8, "high":1e-3, "step":None, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["n_nonzero_coefs"] = trial.suggest_int(f"{self.__class__.__name__}_n_nonzero_coefs", **dict(self.n_nonzero_coefs_space))
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", **dict(self.eps_space))
        set_jitter = trial.suggest_categorical(f"{self.__class__.__name__}_set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float(f"{self.__class__.__name__}_jitter", **dict(self.jitter_space))
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Lars, params)
        self.model = model

        return model
    

@dataclass
class LassoLarsTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    eps_space: Dict[str, Any] = MappingProxyType({"low":np.finfo(float).eps, "high":1e-10, "step":None, "log":True})
    positive_space: Iterable[bool] = (True, False)
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Dict[str, Any] = MappingProxyType({"low":1e-8, "high":1e-3, "step":None, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", **dict(self.eps_space))
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        set_jitter = trial.suggest_categorical(f"{self.__class__.__name__}_set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float(f"{self.__class__.__name__}_jitter", **dict(self.jitter_space))
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", LassoLars, params)
        self.model = model

        return model


@dataclass
class LassoLarsICTuner(BaseTuner):
    criterion_sapce: Iterable[str] = ("aic", "bic")
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    eps_space: Dict[str, Any] = MappingProxyType({"low":np.finfo(float).eps, "high":1e-10, "step":None, "log":True})
    positive_space: Iterable[bool] = (True, False)
    set_noise_variance_space: Iterable[bool] = (True, False)
    noise_variance_space: Dict[str, Any] = MappingProxyType({"low":1e-8, "high":1e-3, "step":None, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_sapce)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", **dict(self.eps_space))
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        set_noise_variance = trial.suggest_categorical(f"{self.__class__.__name__}_set_noise_variance", self.set_noise_variance_space)
        if set_noise_variance:
            params["noise_variance"] = trial.suggest_float(f"{self.__class__.__name__}_noise_variance", **dict(self.noise_variance_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", LassoLarsIC, params)
        self.model = model

        return model
     

@dataclass
class BayesianRidgeTuner(BaseTuner):
    n_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    alpha_1_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    alpha_2_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    lambda_1_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    lambda_2_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    set_alpha_init_space: Iterable[bool] = (True, False)
    alpha_init_space: Iterable[bool] = MappingProxyType({"low":1e-8, "high":1.0, "step":None, "log":True})
    lambda_init_space: Dict[str, Any] = MappingProxyType({"low":1e-8, "high":1.0, "step":None, "log":True})
    compute_score_space: Iterable[bool] = (True, False)
    fit_intercept_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_iter"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter", **dict(self.n_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["alpha_1"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_1", **dict(self.alpha_1_space))
        params["alpha_2"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_2", **dict(self.alpha_2_space))
        params["lambda_1"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_1", **dict(self.lambda_1_space))
        params["lambda_2"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_2", **dict(self.lambda_2_space))
        set_alpha_init = trial.suggest_categorical(f"{self.__class__.__name__}_set_alpha_init", self.set_alpha_init_space)
        if set_alpha_init:
            params["alpha_init"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_init", **dict(self.alpha_init_space))
        params["lambda_init"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_init", **dict(self.lambda_init_space))
        params["compute_score"] = trial.suggest_categorical(f"{self.__class__.__name__}_compute_score", self.compute_score_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", BayesianRidge, params)
        self.model = model

        return model
    

@dataclass
class ARDRegressionTuner(BaseTuner):
    n_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    alpha_1_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    alpha_2_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    lambda_1_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    lambda_2_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    threshold_lambda_space: Dict[str, Any] = MappingProxyType({"low":1e3, "high":1e5, "step":None, "log":True})
    compute_score_space: Iterable[bool] = (True, False)
    fit_intercept_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_iter"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter", **dict(self.n_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["alpha_1"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_1", **dict(self.alpha_1_space))
        params["alpha_2"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_2", **dict(self.alpha_2_space))
        params["lambda_1"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_1", **dict(self.lambda_1_space))
        params["lambda_2"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_2", **dict(self.lambda_2_space))        
        params["threshold_lambda"] = trial.suggest_float(f"{self.__class__.__name__}_threshold_lambda", **dict(self.threshold_lambda_space))
        params["compute_score"] = trial.suggest_categorical(f"{self.__class__.__name__}_compute_score", self.compute_score_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", ARDRegression, params)
        self.model = model
        return model
    

@dataclass
class OrthogonalMatchingPursuitTuner(BaseTuner):
    set_nonzero_coefs_space: Iterable[bool] = (True, False)
    n_nonzero_coefs_space: Dict[str, Any] = MappingProxyType({"low":1, "high":500, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        set_nonzero_coefs = trial.suggest_categorical(f"{self.__class__.__name__}_set_nonzero_coefs", self.set_nonzero_coefs_space)
        if set_nonzero_coefs:
            params["n_nonzero_coefs"] = trial.suggest_int(f"{self.__class__.__name__}_n_nonzero_coefs", **dict(self.n_nonzero_coefs_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", OrthogonalMatchingPursuit, params)
        self.model = model

        return model


@dataclass
class PassiveAggressiveRegressorTuner(BaseTuner):
    C_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":False})
    shuffle_space: Iterable[bool] = (True, False)
    loss_space: Iterable[str] = ("epsilon_insensitive", )
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    epsilon_space: Dict[str, Any] = MappingProxyType({"low":0.05, "high":0.5, "step":None, "log":True})
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", **dict(self.C_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)

        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", **dict(self.epsilon_space))
        params["average"] = trial.suggest_categorical(f"{self.__class__.__name__}_average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", PassiveAggressiveRegressor, params)
        self.model = model

        return model
    

@dataclass
class QuantileRegressorTuner(BaseTuner):
    quantile_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.01, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("highs-ds", "highs-ipm", "highs", "revised simplex")
    solver_options_space: Iterable[Optional[Dict[str, Any]]] = (None, )
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["quantile"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.quantile_space))
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["solver_options"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver_options", self.solver_options_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", QuantileRegressor, params)
        self.model = model

        return model
    

@dataclass
class SGDRegressorTuner(BaseTuner):
    loss_space: Iterable[str] = (
        "squared_error", 
        "huber", 
        "epsilon_insensitive", 
        "squared_epsilon_insensitive")
    penalty_space: Iterable[str] = ("l1", "l2", "elasticnet", None)
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    shuffle_space: Iterable[bool] = (True, False)
    epsilon_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    learning_rate_space: Iterable[str] = ("constant", "optimal", "invscaling", "adaptive")
    eta0_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    power_t_space: Dict[str, Any] = MappingProxyType({"low":-1.0, "high":1.0, "step":None, "log":False})
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":False})
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", **dict(self.epsilon_space))

        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        
        params["learning_rate"] = trial.suggest_categorical(f"{self.__class__.__name__}_learning_rate", self.learning_rate_space)
        params["eta0"] = trial.suggest_float(f"{self.__class__.__name__}_eta0", **dict(self.eta0_space))
        params["power_t"] = trial.suggest_float(f"{self.__class__.__name__}_power_t", **dict(self.power_t_space))
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["average"] = trial.suggest_categorical(f"{self.__class__.__name__}_average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", SGDRegressor, params)
        self.model = model

        return model
    

@dataclass
class PoissonRegressorTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", PoissonRegressor, params)
        self.model = model

        return model


@dataclass
class GammaRegressorTuner(PoissonRegressorTuner):

    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(GammaRegressorTuner, self).sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(PoissonRegressorTuner, self).sample_model(trial)

        params = self.sample_params(trial)
        model = super(PoissonRegressorTuner, self).evaluate_sampled_model("regression", GammaRegressor, params)
        self.model = model
        return model
    

@dataclass
class TweedieRegressorTuner(BaseTuner):
    power_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":3.0, "step":None, "log":True})
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    link_space: Iterable[str] = ("auto", "identity", "log")
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["power"] = trial.suggest_float(f"{self.__class__.__name__}_power", **dict(self.power_space))
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["link"] = trial.suggest_categorical(f"{self.__class__.__name__}_link", self.link_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", TweedieRegressor, params)
        self.model = model

        return model


@dataclass
class HuberRegressorTuner(BaseTuner):
    epsilon_space: Dict[str, Any] = MappingProxyType({"low":1.0, "high":10.0, "step":None, "log":True})
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    fit_intercept_space: Iterable[bool] = (True, False)
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", **dict(self.epsilon_space))
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", HuberRegressor, params)
        self.model = model
        return model
    

@dataclass
class TheilSenRegressorTuner(BaseTuner):
    fit_intercept_space: Iterable[bool] = (True, False)
    max_subpopulation_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1e5, "step":1, "log":True})
    set_n_subsamples_space: Iterable[bool] = (False, )
    n_subsamples_space: Optional[Dict[str, Any]] = MappingProxyType({"low":1, "high":40, "step":1, "log":True})
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":300, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_subpopulation"] = trial.suggest_int(f"{self.__class__.__name__}_max_subpopulation", **dict(self.max_subpopulation_space))
        
        set_n_subsamples = trial.suggest_categorical("set_n_subsamples", self.set_n_subsamples_space)
        if set_n_subsamples:
            params["n_subsamples"] = trial.suggest_int(f"{self.__class__.__name__}_n_subsamples", **dict(self.n_subsamples_space))
        
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", TheilSenRegressor, params)
        self.model = model
        return model    


@dataclass
class RANSACRegressorTuner(BaseTuner):
    estimator: Optional[Union[RegressorMixin, BaseEstimator]] = None
    min_samples_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    residual_threshold_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    max_trials_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    max_skips_space: Dict[str, Any] = MappingProxyType({"low":1, "high":1e5, "step":1, "log":True})
    stop_n_inliers_space: Dict[str, Any] = MappingProxyType({"low":1, "high":1e5, "step":1, "log":True})
    stop_score_space: Dict[str, Any] = MappingProxyType({"low":1.0, "high":1e5, "step":None, "log":True})
    stop_probability_space: Dict[str, Any] = MappingProxyType({"low":0.5, "high":0.99, "step":None, "log":False})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["estimator"] = self.estimator
        params["min_samples"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples", **dict(self.min_samples_space))
        params["residual_threshold"] = trial.suggest_float(f"{self.__class__.__name__}_residual_threshold", **dict(self.residual_threshold_space))
        params["max_trials"] = trial.suggest_int(f"{self.__class__.__name__}_max_trials", **dict(self.max_trials_space))
        params["max_skips"] = trial.suggest_int(f"{self.__class__.__name__}_max_skips", **dict(self.max_skips_space))
        params["stop_n_inliers"] = trial.suggest_int(f"{self.__class__.__name__}_stop_n_inliers", **dict(self.stop_n_inliers_space))
        params["stop_score"] = trial.suggest_float(f"{self.__class__.__name__}_stop_score", **dict(self.stop_score_space))
        params["stop_probability"] = trial.suggest_float(f"{self.__class__.__name__}_stop_probability", **dict(self.stop_probability_space))
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", RANSACRegressor, params)
        self.model = model
        return model    


tuner_model_class_dict: Dict[str, Callable] = {
    LinearRegressionTuner.__name__: LinearRegression,
    LassoTuner.__name__: Lasso,
    RidgeTuner.__name__: Ridge,
    ElasticNetTuner.__name__: ElasticNet,
    MultiTaskLassoTuner.__name__: MultiTaskLasso,
    MultiTaskElasticNetTuner.__name__: MultiTaskElasticNet,
    LarsTuner.__name__: Lars,
    LassoLarsTuner.__name__: LassoLars,
    LassoLarsICTuner.__name__: LassoLarsIC,
    PassiveAggressiveRegressorTuner.__name__: PassiveAggressiveRegressor,
    QuantileRegressorTuner.__name__: QuantileRegressor,
    SGDRegressorTuner.__name__: SGDRegressor,
    BayesianRidgeTuner.__name__: BayesianRidge,
    OrthogonalMatchingPursuitTuner.__name__: OrthogonalMatchingPursuit,
    PoissonRegressorTuner.__name__: PoissonRegressor,
    GammaRegressorTuner.__name__: GammaRegressor,
    TweedieRegressorTuner.__name__: TweedieRegressor,
    HuberRegressorTuner.__name__: HuberRegressor,
    TheilSenRegressorTuner.__name__: TheilSenRegressor,
    ARDRegressionTuner.__name__: ARDRegression,
    RANSACRegressorTuner.__name__: RANSACRegressor
}
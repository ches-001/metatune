import numpy as np
from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Any, Union, Callable
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
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)

        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Lasso, params)
        self.model = model

        return model

 
@dataclass
class RidgeTuner(BaseTuner):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    solver_space: Iterable[str] = ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs")
    positive_space: Iterable[bool] = (True, False)
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)

        if params["solver"] in ["sag", "saga"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)  
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Ridge, params)
        self.model = model

        return model
    

@dataclass
class ElasticNetTuner(BaseTuner):
    alpha_space: Iterable[float] = (0.01, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[Union[bool, Iterable]] = (True, False, )
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    positive_space: Iterable[bool] = (True, False)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", *self.l1_ratio_space, log=False)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", ElasticNet, params)
        self.model = model

        return model
    

@dataclass
class MultiTaskLassoTuner(BaseTuner):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Iterable[int] = (0, 10000)
    is_multitask: str = field(init=False, default=True)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", MultiTaskLasso, params, is_multitask=self.is_multitask)
        self.model = model

        return model
    

@dataclass
class MultiTaskElasticNetTuner(BaseTuner):
    alpha_space: Iterable[float] = (0.01, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    selection_space: Iterable[str] = ("cyclic", "random")
    random_state_space: Iterable[int] = (0, 10000)
    is_multitask: str = field(init=False, default=True)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["selection"] = trial.suggest_categorical(f"{self.__class__.__name__}_selection", self.selection_space)
        if params["selection"] == "random":
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
            
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
    n_nonzero_coefs_space: Iterable[int] = (1, 500)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Iterable[float] = (1e-8, 1e-3)
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["n_nonzero_coefs"] = trial.suggest_int(f"{self.__class__.__name__}_n_nonzero_coefs", *self.n_nonzero_coefs_space, log=False)
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", *self.eps_space, log=False)
        set_jitter = trial.suggest_categorical(f"{self.__class__.__name__}_set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float(f"{self.__class__.__name__}_jitter", *self.jitter_space, log=False)
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", Lars, params)
        self.model = model

        return model
    

@dataclass
class LassoLarsTuner(BaseTuner):
    alpha_space: Iterable[float] = (0.1, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 1000)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    positive_space: Iterable[bool] = (True, False)
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Iterable[float] = (1e-8, 1e-3)
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", *self.eps_space, log=False)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        set_jitter = trial.suggest_categorical(f"{self.__class__.__name__}_set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float(f"{self.__class__.__name__}_jitter", *self.jitter_space, log=False)
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

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
    max_iter_space: Iterable[int] = (100, 1000)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    positive_space: Iterable[bool] = (True, False)
    set_noise_variance_space: Iterable[bool] = (True, False)
    noise_variance_space: Iterable[float] = (1e-8, 1e-3)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical(f"{self.__class__.__name__}_criterion", self.criterion_sapce)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical(f"{self.__class__.__name__}_precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["eps"] = trial.suggest_float(f"{self.__class__.__name__}_eps", *self.eps_space, log=False)
        params["positive"] = trial.suggest_categorical(f"{self.__class__.__name__}_positive", self.positive_space)
        set_noise_variance = trial.suggest_categorical(f"{self.__class__.__name__}_set_noise_variance", self.set_noise_variance_space)
        if set_noise_variance:
            params["noise_variance"] = trial.suggest_float(f"{self.__class__.__name__}_noise_variance", *self.noise_variance_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", LassoLarsIC, params)
        self.model = model

        return model
     

@dataclass
class BayesianRidgeTuner(BaseTuner):
    n_iter_space: Iterable[int] = (100, 1000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    alpha_1_space: Iterable[float] = (1e-6, 1e-3)
    alpha_2_space: Iterable[float] = (1e-6, 1e-3)
    lambda_1_space: Iterable[float] = (1e-6, 1e-3)
    lambda_2_space: Iterable[float] = (1e-6, 1e-3)
    set_alpha_init_space: Iterable[bool] = (True, False)
    alpha_init_space: Iterable[bool] = (1e-8, 1)
    lambda_init_space: Iterable[float] = (1e-8, 1)
    compute_score_space: Iterable[bool] = (True, False)
    fit_intercept_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_iter"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter", *self.n_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["alpha_1"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_1", *self.alpha_1_space, log=False)
        params["alpha_2"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_2", *self.alpha_2_space, log=False)
        params["lambda_1"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_1", *self.lambda_1_space, log=False)
        params["lambda_2"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_2", *self.lambda_2_space, log=False)
        set_alpha_init = trial.suggest_categorical(f"{self.__class__.__name__}_set_alpha_init", self.set_alpha_init_space)
        if set_alpha_init:
            params["alpha_init"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_init", *self.alpha_init_space, log=False)
        params["lambda_init"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_init", *self.lambda_init_space, log=False)
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
    n_iter_space: Iterable[int] = (100, 1000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    alpha_1_space: Iterable[float] = (1e-6, 1e-3)
    alpha_2_space: Iterable[float] = (1e-6, 1e-3)
    lambda_1_space: Iterable[float] = (1e-6, 1e-3)
    lambda_2_space: Iterable[float] = (1e-6, 1e-3)
    threshold_lambda_space: Iterable[int] = (1e3, 1e5)
    compute_score_space: Iterable[bool] = (True, False)
    fit_intercept_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["n_iter"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter", *self.n_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["alpha_1"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_1", *self.alpha_1_space, log=False)
        params["alpha_2"] = trial.suggest_float(f"{self.__class__.__name__}_alpha_2", *self.alpha_2_space, log=False)
        params["lambda_1"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_1", *self.lambda_1_space, log=False)
        params["lambda_2"] = trial.suggest_float(f"{self.__class__.__name__}_lambda_2", *self.lambda_2_space, log=False)        
        params["threshold_lambda"] = trial.suggest_float(f"{self.__class__.__name__}_threshold_lambda", *self.threshold_lambda_space, log=False)
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
    n_nonzero_coefs_space: Iterable[int] = (1, 500)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        set_nonzero_coefs = trial.suggest_categorical(f"{self.__class__.__name__}_set_nonzero_coefs", self.set_nonzero_coefs_space)
        if set_nonzero_coefs:
            params["n_nonzero_coefs"] = trial.suggest_int(f"{self.__class__.__name__}_n_nonzero_coefs", *self.n_nonzero_coefs_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
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
    C_space: Iterable[float] = (0.9, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    shuffle_space: Iterable[bool] = (True, False)
    loss_space: Iterable[str] = ("epsilon_insensitive", )
    random_state_space: Iterable[int] = (0, 10000)
    epsilon_space: Iterable[float] = (0.05, 0.5)
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)

        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)

        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.epsilon_space, log=False)
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
    quantile_space: Iterable[float] = (0.1, 1.0)
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("highs-ds", "highs-ipm", "highs", "revised simplex")
    solver_options_space: Iterable[Optional[Dict[str, Any]]] = (None, )
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["quantile"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.quantile_space, log=False)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
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
    alpha_space: Iterable[float] = (1e-5, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    shuffle_space: Iterable[bool] = (True, False)
    epsilon_space: Iterable[float] = (0.1, 1.0)
    random_state_space: Iterable[int] = (0, 10000)
    learning_rate_space: Iterable[str] = ("constant", "optimal", "invscaling", "adaptive")
    eta0_space: Iterable[float] = (0.1, 1.0)
    power_t_space: Iterable[float] = (-1.0, 1.0)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.epsilon_space, log=False)

        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
        params["learning_rate"] = trial.suggest_categorical(f"{self.__class__.__name__}_learning_rate", self.learning_rate_space)
        params["eta0"] = trial.suggest_float(f"{self.__class__.__name__}_eta0", *self.eta0_space, log=False)
        params["power_t"] = trial.suggest_float(f"{self.__class__.__name__}_power_t", *self.power_t_space, log=False)
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", *self.n_iter_no_change_space, log=False)
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
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)

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
    power_space: Iterable[float] = (0.0, 3.0)
    alpha_space: Iterable[float] = (0.7, 100.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    link_space: Iterable[str] = ("auto", "identity", "log")
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Iterable[int] = (100, 1000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["power"] = trial.suggest_float(f"{self.__class__.__name__}_power", *self.power_space, log=False)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["link"] = trial.suggest_categorical(f"{self.__class__.__name__}_link", self.link_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", TweedieRegressor, params)
        self.model = model

        return model


@dataclass
class HuberRegressorTuner(BaseTuner):
    epsilon_space: Iterable[float] = (1.0, 1e4)
    max_iter_space: Iterable[int] = (100, 1000)
    alpha_space: Iterable[float] = (0.0, 10.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.epsilon_space, log=False)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
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
    max_subpopulation_space: Iterable[int] = (100, 1e4)
    n_subsamples_space: Optional[Iterable[int]] = None
    max_iter_space: Iterable[int] = (100, 300)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    random_state_space: Iterable[int] = (0, 10000)
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_subpopulation"] = trial.suggest_int(f"{self.__class__.__name__}_max_subpopulation", *self.max_subpopulation_space, log=False)
        
        if self.n_subsamples_space:
            params["n_subsamples"] = trial.suggest_int(f"{self.__class__.__name__}_n_subsamples", *self.n_subsamples_space, log=False)
        
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
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
    min_samples_space: Iterable[float] = (0, 1)
    residual_threshold_space: Iterable[float] = (0, 1)
    max_trials_space: Iterable[int] = (100, 1000)
    max_skips_space: Iterable[int] = (1, 1e5)
    stop_n_inliers_space: Iterable[int] = (1, 1e5)
    stop_score_space: Iterable[float] = (1.0, 1e5)
    stop_probability_space: Iterable[float] = (0.5, 0.99)
    random_state_space: Iterable[int] = (0, 10000)
    model: Any = None
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["estimator"] = self.estimator
        params["min_samples"] = trial.suggest_float(f"{self.__class__.__name__}_min_samples", *self.min_samples_space, log=False)
        params["residual_threshold"] = trial.suggest_float(f"{self.__class__.__name__}_residual_threshold", *self.residual_threshold_space, log=False)
        params["max_trials"] = trial.suggest_int(f"{self.__class__.__name__}_max_trials", *self.max_trials_space, log=False)
        params["max_skips"] = trial.suggest_int(f"{self.__class__.__name__}_max_skips", *self.max_skips_space, log=False)
        params["stop_n_inliers"] = trial.suggest_int(f"{self.__class__.__name__}_stop_n_inliers", *self.stop_n_inliers_space, log=False)
        params["stop_score"] = trial.suggest_float(f"{self.__class__.__name__}_stop_score", *self.stop_score_space, log=False)
        params["stop_probability"] = trial.suggest_float(f"{self.__class__.__name__}_stop_probability", *self.stop_probability_space, log=False)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
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
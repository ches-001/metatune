import numpy as np
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
        model = super()._evaluate_sampled_model("regression", MultiTaskLasso, params, is_multitask=self.is_multitask)
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
            "regression", MultiTaskElasticNet, params, is_multitask=self.is_multitask)
        self.model = model

        return model
    

@dataclass
class LarsTuner(SampleClassMixin):
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    n_nonzero_coefs_space: Iterable[int] = (1, 500)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Iterable[float] = (1e-8, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical("precompute", self.precompute_space)
        params["n_nonzero_coefs"] = trial.suggest_int("n_nonzero_coefs", *self.n_nonzero_coefs_space, log=False)
        params["eps"] = trial.suggest_float("eps", *self.eps_space, log=False)
        set_jitter = trial.suggest_categorical("set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float("jitter", *self.jitter_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", Lars, params)
        self.model = model

        return model
    

@dataclass
class LassoLarsTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.1, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 1000)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    positive_space: Iterable[bool] = (True, False)
    set_jitter_space: Iterable[bool] = (True, False)
    jitter_space: Iterable[float] = (1e-8, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical("precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["eps"] = trial.suggest_float("eps", *self.eps_space, log=False)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)
        set_jitter = trial.suggest_categorical("set_jitter", self.set_jitter_space)
        if set_jitter:
            params["jitter"] = trial.suggest_float("jitter", *self.jitter_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", LassoLars, params)
        self.model = model

        return model


@dataclass
class LassoLarsICTuner(SampleClassMixin):
    criterion_sapce: Iterable[str] = ("aic", "bic")
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 1000)
    eps_space: Iterable[float] = (np.finfo(float).eps, 1e-10)
    positive_space: Iterable[bool] = (True, False)
    set_noise_variance_space: Iterable[bool] = (True, False)
    noise_variance_space: Iterable[float] = (1e-8, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical("criterion", self.criterion_sapce)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical("precompute", self.precompute_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["eps"] = trial.suggest_float("eps", *self.eps_space, log=False)
        params["positive"] = trial.suggest_categorical("positive", self.positive_space)
        set_noise_variance = trial.suggest_categorical("set_noise_variance", self.set_noise_variance_space)
        if set_noise_variance:
            params["noise_variance"] = trial.suggest_float("noise_variance", *self.noise_variance_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", LassoLarsIC, params)
        self.model = model

        return model
     

@dataclass
class BayesianRidgeTuner(SampleClassMixin):
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
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["n_iter"] = trial.suggest_int("n_iter", *self.n_iter_space, log=False)
        params["alpha_1"] = trial.suggest_float("alpha_1", *self.alpha_1_space, log=False)
        params["alpha_2"] = trial.suggest_float("tol", *self.alpha_2_space, log=False)
        params["lambda_1"] = trial.suggest_float("tol", *self.lambda_1_space, log=False)
        params["lambda_2"] = trial.suggest_float("tol", *self.lambda_2_space, log=False)
        set_alpha_init = trial.suggest_categorical("set_alpha_init", self.set_alpha_init_space)
        if set_alpha_init:
            params["alpha_init"] = trial.suggest_float("alpha_init", *self.alpha_init_space, log=False)
        params["lambda_init"] = trial.suggest_float("lambda_init", *self.lambda_init_space, log=False)
        params["compute_score"] = trial.suggest_categorical("compute_score", self.compute_score_space)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", BayesianRidge, params)
        self.model = model

        return model
    

@dataclass
class OrthogonalMatchingPursuitTuner(SampleClassMixin):
    set_nonzero_coefs_space: Iterable[bool] = (True, False)
    n_nonzero_coefs_space: Iterable[int] = (1, 500)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    fit_intercept_space: Iterable[bool] = (True, False)
    precompute_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        set_nonzero_coefs = trial.suggest_categorical("set_nonzero_coefs", self.set_nonzero_coefs_space)
        if set_nonzero_coefs:
            params["n_nonzero_coefs"] = trial.suggest_int("n_nonzero_coefs", *self.n_nonzero_coefs_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["precompute"] = trial.suggest_categorical("precompute", self.precompute_space)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", OrthogonalMatchingPursuit, params)
        self.model = model

        return model


@dataclass
class PassiveAggressiveRegressorTuner(SampleClassMixin):
    C_space: Iterable[float] = (0.9, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    shuffle_space: Iterable[bool] = (True, False)
    loss_space: Iterable[str] = ("epsilon_insensitive", )
    epsilon_space: Iterable[float] = (0.05, 0.5)
    average_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["early_stopping"] = trial.suggest_categorical("early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float("validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["shuffle"] = trial.suggest_categorical("shuffle", self.shuffle_space)
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["epsilon"] = trial.suggest_float("epsilon", *self.epsilon_space, log=False)
        params["average"] = trial.suggest_categorical("average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", PassiveAggressiveRegressor, params)
        self.model = model

        return model
    

@dataclass
class SGDRegressorTuner(SampleClassMixin):
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
    learning_rate_space: Iterable[str] = ("constant", "optimal", "invscaling", "adaptive")
    eta0_space: Iterable[float] = (0.1, 1.0)
    power_t_space: Iterable[float] = (-1.0, 1.0)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    average_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["penalty"] = trial.suggest_categorical("penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["shuffle"] = trial.suggest_categorical("shuffle", self.shuffle_space)
        params["epsilon"] = trial.suggest_float("epsilon", *self.epsilon_space, log=False)
        params["learning_rate"] = trial.suggest_categorical("learning_rate", self.learning_rate_space)
        params["eta0"] = trial.suggest_float("eta0", *self.eta0_space, log=False)
        params["power_t"] = trial.suggest_float("power_t", *self.power_t_space, log=False)
        params["early_stopping"] = trial.suggest_categorical("early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float("validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["average"] = trial.suggest_categorical("average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", SGDRegressor, params)
        self.model = model

        return model
    

@dataclass
class PoissonRegressorTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.01, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["solver"] = trial.suggest_categorical("solver", self.solver_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", PoissonRegressor, params)
        self.model = model

        return model


@dataclass
class GammaRegressorTuner(PoissonRegressorTuner):

    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        return super(GammaRegressorTuner, self)._sample_params(trial)
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super(PoissonRegressorTuner, self).model(trial)

        params = self._sample_params(trial)
        model = super(PoissonRegressorTuner, self)._evaluate_sampled_model("regression", GammaRegressor, params)
        self.model = model
        return model
    

@dataclass
class TweedieRegressorTuner(SampleClassMixin):
    power_space: Iterable[float] = (0.0, 3.0)
    alpha_space: Iterable[float] = (0.7, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    link_space: Iterable[str] = ("auto", "identity", "log")
    solver_space: Iterable[str] = ("lbfgs", "newton-cholesky")
    max_iter_space: Iterable[int] = (100, 1000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["power"] = trial.suggest_float("power", *self.power_space, log=False)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["link"] = trial.suggest_categorical("link", self.link_space)
        params["solver"] = trial.suggest_categorical("solver", self.solver_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", TweedieRegressor, params)
        self.model = model

        return model


@dataclass
class HuberRegressorTuner(SampleClassMixin):
    epsilon_space: Iterable[float] = (1.0, 1e4)
    max_iter_space: Iterable[int] = (100, 1000)
    alpha_space: Iterable[float] = (0.0, 10.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["epsilon"] = trial.suggest_float("epsilon", *self.epsilon_space, log=False)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", HuberRegressor, params)
        self.model = model
        return model
    

@dataclass
class QuantileRegressorTuner(SampleClassMixin):
    quantile_space: Iterable[float] = (0.0, 1.0)
    alpha_space: Iterable[float] = (0.0, 10.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    solver_space: Iterable[str] = ("highs-ds", "highs-ipm", "highs", "interior-point", "revised simplex")
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["quantile"] = trial.suggest_float("quantile", *self.quantile_space, log=False)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["solver"] = trial.suggest_categorical("solver", self.solver_space)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("regression", QuantileRegressor, params)
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
    SGDRegressorTuner.__name__: SGDRegressor,
    BayesianRidgeTuner.__name__: BayesianRidge,
    OrthogonalMatchingPursuitTuner.__name__: OrthogonalMatchingPursuit,
    PoissonRegressorTuner.__name__: PoissonRegressor,
    GammaRegressorTuner.__name__: GammaRegressor,
    TweedieRegressorTuner.__name__: TweedieRegressor,
}
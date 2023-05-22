from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.linear_model import LinearRegression, Lasso, Ridge

@dataclass
class LinearRegressionModel(SampleClassMixin):
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
        model = LinearRegression(**params)
        
        self.model = model
        return model
    

@dataclass
class LassoModel(SampleClassMixin):
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
        model = Lasso(**params)
        
        self.model = model
        return model

 
@dataclass
class RidgeModel(SampleClassMixin):
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
        
        non_positive_solvers: Iterable[str] = ["cholesky", "sparse_cg", "sag", "saga", "svd", "lsqr"]

        if params["solver"] in non_positive_solvers:
            params["positive"] = False

        elif params["solver"] == "lbfgs":
            params["positive"] = True

        trial.set_user_attr("positive", params["positive"])
            
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = Ridge(**params)
        
        self.model = model
        return model
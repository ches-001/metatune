from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.svm import SVR, LinearSVR

@dataclass
class SVRModel(SampleClassMixin):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    shrinking_space: Iterable[bool] = (True, )
    epsilon_space: Iterable[float] = (0.1, 0.5)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["kernel"] = trial.suggest_categorical("kernel", self.kernel_space)
        params["degree"] = trial.suggest_int("degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical("gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float("coef0", *self.coef0_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["shrinking"] = trial.suggest_categorical("shrinking", self.shrinking_space)
        params["epsilon"] = trial.suggest_float("epsilon", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = SVR(**params)
        
        self.model = model
        return model


@dataclass
class LinearSVRModel(SampleClassMixin):
    epsilon_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    loss_space: Iterable[str] = ("epsilon_insensitive", "squared_epsilon_insensitive")
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Iterable[float] = (0.5, 1.0)
    dual_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (500, 2000)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["epsilon"] = trial.suggest_float("epsilon", *self.tol_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float("intercept_scaling", *self.intercept_scaling_space, log=False)
        params["dual"] = trial.suggest_categorical("dual", self.dual_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)

        if params["loss"] == "epsilon_insensitive":
            params["dual"] = True
            trial.set_user_attr("dual", params["dual"])
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = LinearSVR(**params)
        
        self.model = model
        return model
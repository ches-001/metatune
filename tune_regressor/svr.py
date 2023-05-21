from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.svm import SVR

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

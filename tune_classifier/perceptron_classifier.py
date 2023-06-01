from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.linear_model import Perceptron


@dataclass
class PerceptronClassifierTuner(BaseTuner):
    penalty_space: Iterable[str] = ("l2", "l1", "elasticnet")
    alpha_space: Dict[str, Any] = MappingProxyType({"low": 0.0001, "high": 1.0, "step": None, "log": True})
    max_iter_space: Dict[str, Any] = MappingProxyType({"low": 100, "high": 1000, "step": 10, "log": False})
    tol_space: Dict[str, Any] = MappingProxyType({"low": 1e-6, "high": 1e-3, "step": None, "log": True})

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", Perceptron, params)

        self.model = model

        return model

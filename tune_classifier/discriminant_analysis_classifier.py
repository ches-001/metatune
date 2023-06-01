from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


@dataclass
class LDAClassifierTuner(BaseTuner):
    solver_space: Iterable[str] = ("svd", "lsqr", "eigen")
    shrinkage_space: Iterable[str] = (None, "auto")
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    priors_space: Iterable[Optional[Iterable[float]]] = (None, )
    store_covariance: Iterable[bool] = (False, )
    # covariance_estimator: Iterable[Callable] = (None, )

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        if self.is_valid_categorical_space(self.shrinkage_space):
            params["shrinkage"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinkage", self.shrinkage_space)
        else:
            params["shrinkage"] = trial.suggest_float(f"{self.__class__.__name__}_shrinkage", **dict(self.shrinkage_space))

        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["priors"] = trial.suggest_categorical(f"{self.__class__.__name__}_prior", self.priors_space)
        params["store_covariance"] = trial.suggest_categorical(f"{self.__class__.__name__}_store_covariance", self.store_covariance)
        # params["covariance_estimator"] = trial.suggest_
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", LinearDiscriminantAnalysis, params)

        self.model = model

        return model


@dataclass
class QDAClassifierTuner(BaseTuner):
    reg_param_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    priors_space: Iterable[Optional[Iterable[float]]] = (None,)
    store_covariance: Iterable[bool] = (False,)
    # covariance_estimator: Iterable[Callable] = (None,)

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["reg_param"] = trial.suggest_float(f"{self.__class__.__name__}_reg_param", **dict(self.reg_param_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["priors"] = trial.suggest_categorical(f"{self.__class__.__name__}_prior", self.priors_space)
        params["store_covariance"] = trial.suggest_categorical(f"{self.__class__.__name__}_store_covariance",  self.store_covariance)
        # params["covariance_estimator"] = trial.suggest_

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = self.evaluate_sampled_model("classification", QuadraticDiscriminantAnalysis, params)

        self.model = model
        return model
from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union, Callable
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


@dataclass
class LDAClassifierTuner(BaseTuner):
    solver_space: Iterable[str] = ("svd", "lsqr", "eigen")
    shrinkage_space: Iterable[str] = (None, "auto")
    tol_space: Iterable[float] = (1e-10, 1e-1)

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        if self.is_valid_categorical_space(self.shrinkage_space):
            params["shrinkage"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinkage", self.shrinkage_space)
        else:
            params["shrinkage"] = trial.suggest_float(f"{self.__class__.__name__}_shrinkage", *self.shrinkage_space)

        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=True)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", LinearDiscriminantAnalysis, params)

        self.model = model

        return model


@dataclass
class QDAClassifierTuner(BaseTuner):
    reg_param_space: Iterable[float] = (0.0, 1.0)
    tol_space: Iterable[float] = (1e-10, 1e-1)

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["reg_param"] = trial.suggest_float(f"{self.__class__.__name__}_reg_param", *self.reg_param_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=True)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = self.evaluate_sampled_model("classification", QuadraticDiscriminantAnalysis, params)

        self.model = model
        return model


tuner_model_class_dict: Dict[str, Callable] = {
    LDAClassifierTuner.__name__: LinearDiscriminantAnalysis,
    QDAClassifierTuner.__name__: QuadraticDiscriminantAnalysis
}
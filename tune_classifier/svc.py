from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.svm import SVC, LinearSVC


@dataclass
class SVCTuner(SampleClassMixin):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    shrinking_space: Iterable[bool] = (True, )
    probability_space: Iterable[bool] = (True, )
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
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["shrinking"] = trial.suggest_categorical("shrinking", self.shrinking_space)
        params["probability"] = trial.suggest_categorical("probability", self.probability_space)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", SVC, params)
        return model
    

@dataclass
class LinearSVCTuner(SampleClassMixin):
    penalty_space: Iterable[str] = ("l1", "l2")
    loss_space: Iterable[str] = ("hinge", "squared_hinge")
    dual_space: Iterable[bool] = (True, False)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    multi_class_space: Iterable[str] = ("ovr", "crammer_singer")
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Iterable[float] = (0.5, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    max_iter_space: Iterable[int] = (500, 2000)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical("penalty", self.penalty_space)
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["dual"] = trial.suggest_categorical("dual", self.dual_space)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["multi_class"] = trial.suggest_categorical("multi_class", self.multi_class_space)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float("intercept_scaling", *self.intercept_scaling_space, log=False)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", LinearSVC, params)
        self.model = model

        return model


tuner_model_class_dict: Dict[str, Callable] = {
    SVCTuner.__name__: SVC,
    LinearSVCTuner.__name__: LinearSVC
}

from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.svm import SVC, LinearSVC, NuSVC


@dataclass
class SVCTuner(SampleClassMixin):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.5, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    shrinking_space: Iterable[bool] = (True, )
    probability_space: Iterable[bool] = (True, )
    random_state_space: Iterable[int] = (0, 10000)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", *self.coef0_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["probability"] = trial.suggest_categorical(f"{self.__class__.__name__}_probability", self.probability_space)
        if params["probability"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
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
    C_space: Iterable[float] = (0.5, 1.0)
    multi_class_space: Iterable[str] = ("ovr", "crammer_singer")
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Iterable[float] = (0.5, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    max_iter_space: Iterable[int] = (500, 2000)
    random_state_space: Iterable[int] = (0, 10000)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["dual"] = trial.suggest_categorical(f"{self.__class__.__name__}_dual", self.dual_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["multi_class"] = trial.suggest_categorical(f"{self.__class__.__name__}_multi_class", self.multi_class_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float(f"{self.__class__.__name__}_intercept_scaling", *self.intercept_scaling_space, log=False)
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        if params["dual"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", LinearSVC, params)
        self.model = model

        return model
    

@dataclass
class NuSVCTuner(SampleClassMixin):
    nu_space: Iterable[float] = (0.1, 0.5)
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    shrinking_space: Iterable[bool] = (True, )
    probability_space: Iterable[bool] = (True, )
    tol_space: Iterable[float] = (1e-6, 1e-3)
    class_weight_space: Iterable[str] = ("balanced", )
    decision_function_shape_space: Iterable[str] = ("ovo", "ovr")
    break_ties_space: Iterable[bool] = (False, )
    random_state_space: Iterable[int] = (0, 10000)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        params["nu"] = trial.suggest_float(f"{self.__class__.__name__}_nu", *self.nu_space, log=False)
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", *self.coef0_space, log=False)
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["probability"] = trial.suggest_categorical(f"{self.__class__.__name__}_probability", self.probability_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["decision_function_shape"] = trial.suggest_categorical(f"{self.__class__.__name__}_decision_function_shape", self.decision_function_shape_space)
        params["break_ties"] = trial.suggest_categorical(f"{self.__class__.__name__}_break_ties", self.break_ties_space)

        if params["probability"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", NuSVC, params)
        return model


tuner_model_class_dict: Dict[str, Callable] = {
    SVCTuner.__name__: SVC,
    LinearSVCTuner.__name__: LinearSVC,
    NuSVCTuner.__name__: NuSVC,
}

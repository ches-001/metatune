from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.svm import SVC, LinearSVC, NuSVC
from types import MappingProxyType


@dataclass
class SVCTuner(BaseTuner):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Dict[str, Any] = MappingProxyType({"low":1, "high":5, "step":1, "log":False})
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":0.5, "step":None, "log":False})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    C_space: Dict[str, Any] = MappingProxyType({"low":0.5, "high":1.0, "step":None, "log":False})
    class_weight_space: Iterable[str] = ("balanced", )
    shrinking_space: Iterable[bool] = (True, )
    probability_space: Iterable[bool] = (True, )
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", **dict(self.degree_space))
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", **dict(self.coef0_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", **dict(self.C_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["probability"] = trial.suggest_categorical(f"{self.__class__.__name__}_probability", self.probability_space)
        if params["probability"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", SVC, params)
        return model
    

@dataclass
class LinearSVCTuner(BaseTuner):
    penalty_space: Iterable[str] = ("l1", "l2")
    loss_space: Iterable[str] = ("hinge", "squared_hinge")
    dual_space: Iterable[bool] = (True, False)
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    C_space: Dict[str, Any] = MappingProxyType({"low":0.5, "high":1.0, "step":None, "log":False})
    multi_class_space: Iterable[str] = ("ovr", "crammer_singer")
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Dict[str, Any] = MappingProxyType({"low":0.5, "high":1.0, "step":None, "log":False})
    class_weight_space: Iterable[str] = ("balanced", )
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":500, "high":2000, "step":1, "log":True})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["dual"] = trial.suggest_categorical(f"{self.__class__.__name__}_dual", self.dual_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", **dict(self.C_space))
        params["multi_class"] = trial.suggest_categorical(f"{self.__class__.__name__}_multi_class", self.multi_class_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float(f"{self.__class__.__name__}_intercept_scaling", **dict(self.intercept_scaling_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        if params["dual"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", LinearSVC, params)
        self.model = model

        return model
    

@dataclass
class NuSVCTuner(BaseTuner):
    nu_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Dict[str, Any] = MappingProxyType({"low":1, "high":5, "step":1, "log":False})
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":0.5, "step":None, "log":False})
    shrinking_space: Iterable[bool] = (True, )
    probability_space: Iterable[bool] = (True, )
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    class_weight_space: Iterable[str] = ("balanced", )
    decision_function_shape_space: Iterable[str] = ("ovo", "ovr")
    break_ties_space: Iterable[bool] = (False, )
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["nu"] = trial.suggest_float(f"{self.__class__.__name__}_nu", **dict(self.nu_space))
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", **dict(self.degree_space))
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", **dict(self.coef0_space))
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["probability"] = trial.suggest_categorical(f"{self.__class__.__name__}_probability", self.probability_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["decision_function_shape"] = trial.suggest_categorical(f"{self.__class__.__name__}_decision_function_shape", self.decision_function_shape_space)
        params["break_ties"] = trial.suggest_categorical(f"{self.__class__.__name__}_break_ties", self.break_ties_space)

        if params["probability"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", NuSVC, params)
        return model

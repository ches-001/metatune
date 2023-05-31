from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from types import MappingProxyType
from sklearn.linear_model import (
    LogisticRegression, 
    Perceptron, 
    PassiveAggressiveClassifier,
    SGDClassifier)

@dataclass
class LogisticRegressionTuner(BaseTuner):
    penalty_space: Iterable[Optional[str]] = ("l1", "l2", "elasticnet", None)
    dual_space: Iterable[bool] = (True, False)
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    C_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Dict[str, Any] = MappingProxyType({"low":0.5, "high":1.0, "step":None, "log":False})
    class_weight_space: Iterable[str] = ("balanced", )
    solver_space: Iterable[str] = ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":1000, "step":1, "log":True})
    multi_class_space: Iterable[str] = ("auto", )
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["dual"] = trial.suggest_categorical(f"{self.__class__.__name__}_dual", self.dual_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", **dict(self.C_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float(f"{self.__class__.__name__}_intercept_scaling", **dict(self.intercept_scaling_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["multi_class"] = trial.suggest_categorical(f"{self.__class__.__name__}_multi_class", self.multi_class_space)

        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))

        if params["solver"] in ["sag", "saga", "liblinear"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", LogisticRegression, params)
        self.model = model

        return model
    

@dataclass
class PerceptronTuner(BaseTuner):
    penalty_space: Iterable[Optional[str]] = ("l1", "l2", "elasticnet", None)
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    shuffle_space: Iterable[bool] = (True, False)
    eta0_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    class_weight_space: Iterable[str] = ("balanced", )
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["eta0"] = trial.suggest_float(f"{self.__class__.__name__}_eta0", **dict(self.eta0_space))
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", Perceptron, params)
        self.model = model

        return model


@dataclass
class PassiveAggressiveClassifierTuner(BaseTuner):
    C_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    shuffle_space: Iterable[bool] = (True, False)
    loss_space: Iterable[str] = ("hinge", )
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    class_weight_space: Iterable[str] = ("balanced", )
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", **dict(self.C_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["average"] = trial.suggest_categorical(f"{self.__class__.__name__}_average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", PassiveAggressiveClassifier, params)
        self.model = model

        return model


@dataclass
class SGDClassifierTuner(BaseTuner):
    loss_space: Iterable[str] = (
        "hinge", 
        "log_loss", 
        "modified_huber", 
        "squared_hinge", 
        "perceptron", 
        "squared_error",
        "huber", 
        "epsilon_insensitive", 
        "squared_epsilon_insensitive")
    penalty_space: Iterable[str] = ("l1", "l2", "elasticnet", None)
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1.0, "step":None, "log":True})
    l1_ratio_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":100, "high":2000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-6, "high":1e-3, "step":None, "log":True})
    shuffle_space: Iterable[bool] = (True, False)
    epsilon_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    learning_rate_space: Iterable[str] = ("constant", "optimal", "invscaling", "adaptive")
    eta0_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    power_t_space: Dict[str, Any] = MappingProxyType({"low":-1.0, "high":1.0, "step":None, "log":False})
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":1, "high":100, "step":1, "log":True})
    class_weight_space: Iterable[str] = ("balanced", )
    average_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["penalty"] = trial.suggest_categorical(f"{self.__class__.__name__}_penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["l1_ratio"] = trial.suggest_float(f"{self.__class__.__name__}_l1_ratio", **dict(self.l1_ratio_space))
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", **dict(self.epsilon_space))
        if params["shuffle"]:
            params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))

        params["learning_rate"] = trial.suggest_categorical(f"{self.__class__.__name__}_learning_rate", self.learning_rate_space)
        params["eta0"] = trial.suggest_float(f"{self.__class__.__name__}_eta0", **dict(self.eta0_space))
        params["power_t"] = trial.suggest_float(f"{self.__class__.__name__}_power_t", **dict(self.power_t_space))
        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["class_weight"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_weight", self.class_weight_space)
        params["average"] = trial.suggest_categorical(f"{self.__class__.__name__}_average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", SGDClassifier, params)
        self.model = model

        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    LogisticRegressionTuner.__name__: LogisticRegression,
    PerceptronTuner.__name__: Perceptron,
    PassiveAggressiveClassifierTuner.__name__: PassiveAggressiveClassifier,
    SGDClassifierTuner.__name__: SGDClassifier,
}
from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.linear_model import (
    LogisticRegression, 
    Perceptron, 
    PassiveAggressiveClassifier,
    SGDClassifier)

@dataclass
class LogisticRegressionTuner(SampleClassMixin):
    penalty_space: Iterable[Optional[str]] = ("l1", "l2", "elasticnet", None)
    dual_space: Iterable[bool] = (True, False)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Iterable[float] = (0.5, 1.0)
    class_weight_space: Iterable[str] = ("balanced", )
    solver_space: Iterable[str] = ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")
    max_iter_space: Iterable[int] = (100, 1000)
    multi_class_space: Iterable[str] = ("auto", )
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical("penalty", self.penalty_space)
        params["dual"] = trial.suggest_categorical("dual", self.dual_space)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float("intercept_scaling", *self.intercept_scaling_space, log=False)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["solver"] = trial.suggest_categorical("solver", self.solver_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["multi_class"] = trial.suggest_categorical("multi_class", self.multi_class_space)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", LogisticRegression, params)
        self.model = model

        return model
    

@dataclass
class PerceptronTuner(SampleClassMixin):
    penalty_space: Iterable[Optional[str]] = ("l1", "l2", "elasticnet", None)
    alpha_space: Iterable[float] = (1e-5, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    shuffle_space: Iterable[bool] = (True, False)
    eta0_space: Iterable[float] = (0.1, 1.0)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.0, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    class_weight_space: Iterable[str] = ("balanced", )
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["penalty"] = trial.suggest_categorical("penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["shuffle"] = trial.suggest_categorical("shuffle", self.shuffle_space)
        params["eta0"] = trial.suggest_float("eta0", *self.eta0_space, log=False)
        params["early_stopping"] = trial.suggest_categorical("early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float("validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", Perceptron, params)
        self.model = model

        return model


@dataclass
class PassiveAggressiveClassifierTuner(SampleClassMixin):
    C_space: Iterable[float] = (0.9, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.0, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    shuffle_space: Iterable[bool] = (True, False)
    loss_space: Iterable[str] = ("hinge", )
    class_weight_space: Iterable[str] = ("balanced", )
    average_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["early_stopping"] = trial.suggest_categorical("early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float("validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["shuffle"] = trial.suggest_categorical("shuffle", self.shuffle_space)
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["average"] = trial.suggest_categorical("average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", PassiveAggressiveClassifier, params)
        self.model = model

        return model


@dataclass
class SGDClassifierTuner(SampleClassMixin):
    loss_space: Iterable[str] = (
        "hinge", 
        "log_loss", 
        "log", 
        "modified_huber", 
        "squared_hinge", 
        "perceptron", 
        "squared_error",
        "huber", 
        "epsilon_insensitive", 
        "squared_epsilon_insensitive")
    penalty_space: Iterable[str] = ("l1", "l2", "elasticnet", None)
    alpha_space: Iterable[float] = (1e-5, 1.0)
    l1_ratio_space: Iterable[float] = (0.0, 1.0)
    fit_intercept_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (100, 2000)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    shuffle_space: Iterable[bool] = (True, False)
    epsilon_space: Iterable[float] = (0.1, 1.0)
    learning_rate_space: Iterable[str] = ("constant", "optimal", "invscaling", "adaptive")
    eta0_space: Iterable[float] = (0.1, 1.0)
    power_t_space: Iterable[float] = (-1.0, 1.0)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.0, 0.5)
    n_iter_no_change_space: Iterable[int] = (1, 100)
    class_weight_space: Iterable[str] = ("balanced", )
    average_space: Iterable[bool] = (True, False)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["loss"] = trial.suggest_categorical("loss", self.loss_space)
        params["penalty"] = trial.suggest_categorical("penalty", self.penalty_space)
        params["alpha"] = trial.suggest_float("alpha", *self.alpha_space, log=False)
        params["l1_ratio"] = trial.suggest_float("l1_ratio", *self.l1_ratio_space, log=False)
        params["fit_intercept"] = trial.suggest_categorical("fit_intercept", self.fit_intercept_space)
        params["max_iter"] = trial.suggest_int("max_iter", *self.max_iter_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["shuffle"] = trial.suggest_categorical("shuffle", self.shuffle_space)
        params["epsilon"] = trial.suggest_float("epsilon", *self.epsilon_space, log=False)
        params["learning_rate"] = trial.suggest_categorical("learning_rate", self.learning_rate_space)
        params["eta0"] = trial.suggest_float("eta0", *self.eta0_space, log=False)
        params["power_t"] = trial.suggest_float("power_t", *self.power_t_space, log=False)
        params["early_stopping"] = trial.suggest_categorical("early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float("validation_fraction", *self.validation_fraction_space, log=False)
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", *self.n_iter_no_change_space, log=False)
        params["class_weight"] = trial.suggest_categorical("class_weight", self.class_weight_space)
        params["average"] = trial.suggest_categorical("average", self.average_space)
 
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", SGDClassifier, params)
        self.model = model

        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    LogisticRegressionTuner.__name__: LogisticRegression,
    PerceptronTuner.__name__: Perceptron,
    PassiveAggressiveClassifierTuner.__name__: PassiveAggressiveClassifier,
    SGDClassifierTuner.__name__: SGDClassifier,
}
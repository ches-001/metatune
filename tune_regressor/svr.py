from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.svm import SVR, LinearSVR, NuSVR


@dataclass
class SVRTuner(BaseTuner):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.5, 1.0)
    shrinking_space: Iterable[bool] = (True, )
    epsilon_space: Iterable[float] = (0.1, 0.5)
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", *self.coef0_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", SVR, params)
        self.model = model

        return model


@dataclass
class LinearSVRTuner(BaseTuner):
    epsilon_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.5, 1.0)
    loss_space: Iterable[str] = ("epsilon_insensitive", "squared_epsilon_insensitive")
    fit_intercept_space: Iterable[bool] = (True, False)
    intercept_scaling_space: Iterable[float] = (0.5, 1.0)
    dual_space: Iterable[bool] = (True, False)
    max_iter_space: Iterable[int] = (500, 2000)
    random_state_space: Iterable[int] = (0, 10000)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)
        
        params = {}
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.epsilon_space, log=False)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["loss"] = trial.suggest_categorical(f"{self.__class__.__name__}_loss", self.loss_space)
        params["fit_intercept"] = trial.suggest_categorical(f"{self.__class__.__name__}_fit_intercept", self.fit_intercept_space)
        params["intercept_scaling"] = trial.suggest_float(f"{self.__class__.__name__}_intercept_scaling", *self.intercept_scaling_space, log=False)
        params["dual"] = trial.suggest_categorical(f"{self.__class__.__name__}_dual", self.dual_space)
        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space, log=False)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", LinearSVR, params)
        self.model = model

        return model
    
    
@dataclass
class NuSVRTuner(BaseTuner):
    nu_space: Iterable[float] = (0.1, 1.0)
    C_space: Iterable[float] = (0.5, 1.0)
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    shrinking_space: Iterable[bool] = (True, )
    tol_space: Iterable[float] = (1e-6, 1e-3)
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["nu"] = trial.suggest_float(f"{self.__class__.__name__}_nu", *self.nu_space, log=False)
        params["C"] = trial.suggest_float(f"{self.__class__.__name__}_C", *self.C_space, log=False)
        params["kernel"] = trial.suggest_categorical(f"{self.__class__.__name__}_kernel", self.kernel_space)
        params["degree"] = trial.suggest_int(f"{self.__class__.__name__}_degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical(f"{self.__class__.__name__}_gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float(f"{self.__class__.__name__}_coef0", *self.coef0_space, log=False)
        params["shrinking"] = trial.suggest_categorical(f"{self.__class__.__name__}_shrinking", self.shrinking_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space, log=False)
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("regression", NuSVR, params)
        return model
    

tuner_model_class_dict: Dict[str, Callable] = {
    SVRTuner.__name__: SVR,
    LinearSVRTuner.__name__: LinearSVR,
    NuSVRTuner.__name__: NuSVR,
}

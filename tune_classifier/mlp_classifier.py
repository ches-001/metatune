from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from types import MappingProxyType
from sklearn.neural_network import MLPClassifier


@dataclass
class MLPClassifierTuner(BaseTuner):
    n_hidden_space: Dict[str, Any] = MappingProxyType({"low":1, "high":5, "step":1, "log":False})
    hidden_layer_sizes_space: Dict[str, Any] = MappingProxyType({"low":100, "high":200, "step":1, "log":True})
    activation_space: Iterable[str] = ("identity", "logistic", "tanh", "relu")
    solver_space: Iterable[str] = ("lbfgs", "sgd", "adam")
    alpha_space: Dict[str, Any] = MappingProxyType({"low":1e-4, "high":1.0, "step":None, "log":True})
    batch_size_space: Dict[str, Any] = MappingProxyType({"low":8, "high":256, "step":1, "log":True})
    learning_rate_space: Iterable[str] = ("constant", "invscaling", "adaptive")
    learning_rate_init_space: Dict[str, Any] = MappingProxyType({"low":1e-4, "high":1e-2, "step":None, "log":True})
    power_t_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":1.0, "step":None, "log":False})
    max_iter_space: Dict[str, Any] = MappingProxyType({"low":200, "high":1000, "step":1, "log":True})
    shuffle_space: Iterable[bool] = (True, False)
    random_state_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10000, "step":1, "log":True})
    tol_space: Dict[str, Any] = MappingProxyType({"low":1e-5, "high":1e-2, "step":None, "log":True})
    momentum_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    nesterovs_momentum_space: Iterable[bool] = (True, False)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.5, "step":None, "log":False})
    beta_1_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    beta_2_space: Dict[str, Any] = MappingProxyType({"low":0.9, "high":1.0, "step":None, "log":False})
    epsilon_space: Dict[str, Any] = MappingProxyType({"low":1e-8, "high":1e-5, "step":None, "log":True})
    n_iter_no_change_space: Dict[str, Any] = MappingProxyType({"low":3, "high":50, "step":1, "log":True})
    max_fun_space: Dict[str, Any] = MappingProxyType({"low":10000, "high":20000, "step":1, "log":True})
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        n_hidden = trial.suggest_int(f"{self.__class__.__name__}_n_hidden", **dict(self.n_hidden_space))
        params["hidden_layer_sizes"] = tuple(trial.suggest_int(f"hidden_layer_sizes_{i}", 
                                                          **dict(self.hidden_layer_sizes_space)) 
                                                          for i in range(n_hidden))
        params["activation"] = trial.suggest_categorical(f"{self.__class__.__name__}_activation", self.activation_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["batch_size"] = trial.suggest_int(f"{self.__class__.__name__}_batch_size", **dict(self.batch_size_space))
        params["learning_rate"] = trial.suggest_categorical(f"{self.__class__.__name__}_learning_rate", self.learning_rate_space)
        params["learning_rate_init"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate_init", **dict(self.learning_rate_init_space))

        if params["learning_rate"] == "invscaling" and params["solver"] == "sgd":
            params["power_t"] = trial.suggest_float(f"{self.__class__.__name__}_power_t", **dict(self.power_t_space))

        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", **dict(self.max_iter_space))
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", **dict(self.random_state_space))
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", **dict(self.tol_space))

        if params["solver"] == "sgd":
            params["momentum"] = trial.suggest_float(f"{self.__class__.__name__}_momentum", **dict(self.momentum_space))
            params["nesterovs_momentum"] = trial.suggest_categorical(f"{self.__class__.__name__}_nesterovs_momentum", self.nesterovs_momentum_space)

        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", **dict(self.validation_fraction_space))
        params["beta_1"] = trial.suggest_float(f"{self.__class__.__name__}_beta_1", **dict(self.beta_1_space))
        params["beta_2"] = trial.suggest_float(f"{self.__class__.__name__}_beta_2", **dict(self.beta_2_space))
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", **dict(self.epsilon_space))
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", **dict(self.n_iter_no_change_space))
        params["max_fun"] = trial.suggest_int(f"{self.__class__.__name__}_max_fun", **dict(self.max_fun_space))
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", MLPClassifier, params)
        self.model = model
        return model


tuner_model_class_dict: Dict[str, Callable] = {
    MLPClassifierTuner.__name__: MLPClassifier
}
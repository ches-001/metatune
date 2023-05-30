from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable, Union
from sklearn.neural_network import MLPClassifier


@dataclass
class MLPClassifierTuner(BaseTuner):
    n_hidden_space: Iterable[int] = (1, 5)
    hidden_layer_sizes_space: Iterable[int] = (100, 200)
    activation_space: Iterable[str] = ("identity", "logistic", "tanh", "relu")
    solver_space: Iterable[str] = ("lbfgs", "sgd", "adam")
    alpha_space: Iterable[float] = (1e-4, 1)
    batch_size_space: Iterable[int] = (8, 256)
    learning_rate_space: Iterable[str] = ("constant", "invscaling", "adaptive")
    learning_rate_init_space: Iterable[float] = (1e-4, 1e-2)
    power_t_space: Iterable[float] = (0.1, 1.0)
    max_iter_space: Iterable[int] = (200, 1000)
    shuffle_space: Iterable[bool] = (True, False)
    random_state_space: Iterable[int] = (0, 10000)
    tol_space: Iterable[float] = (1e-5, 1e-2)
    momentum_space: Iterable[float] = (0.9, 1.0)
    nesterovs_momentum_space: Iterable[bool] = (True, False)
    early_stopping_space: Iterable[bool] = (True, False)
    validation_fraction_space: Iterable[float] = (0.1, 0.5)
    beta_1_space: Iterable[float] = (0.9, 1.0)
    beta_2_space: Iterable[float] = (0.9, 1.0)
    epsilon_space: Iterable[float] = (1e-8, 1e-5)
    n_iter_no_change_space: Iterable[int] = (3, 50)
    max_fun_space: Iterable[int] = (10000, 20000)
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        n_hidden = trial.suggest_int(f"{self.__class__.__name__}_n_hidden", *self.n_hidden_space)
        params["hidden_layer_sizes"] = tuple(trial.suggest_int(f"hidden_layer_sizes_{i}", 
                                                          *self.hidden_layer_sizes_space) 
                                                          for i in range(n_hidden))
        params["activation"] = trial.suggest_categorical(f"{self.__class__.__name__}_activation", self.activation_space)
        params["solver"] = trial.suggest_categorical(f"{self.__class__.__name__}_solver", self.solver_space)
        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", *self.alpha_space)
        params["batch_size"] = trial.suggest_int(f"{self.__class__.__name__}_batch_size", *self.batch_size_space)
        params["learning_rate"] = trial.suggest_categorical(f"{self.__class__.__name__}_learning_rate", self.learning_rate_space)
        params["learning_rate_init"] = trial.suggest_float(f"{self.__class__.__name__}_learning_rate_init", *self.learning_rate_init_space)

        if params["learning_rate"] == "invscaling" and params["solver"] == "sgd":
            params["power_t"] = trial.suggest_float(f"{self.__class__.__name__}_power_t", *self.power_t_space)

        params["max_iter"] = trial.suggest_int(f"{self.__class__.__name__}_max_iter", *self.max_iter_space)
        params["shuffle"] = trial.suggest_categorical(f"{self.__class__.__name__}_shuffle", self.shuffle_space)
        params["random_state"] = trial.suggest_int(f"{self.__class__.__name__}_random_state", *self.random_state_space)
        params["tol"] = trial.suggest_float(f"{self.__class__.__name__}_tol", *self.tol_space)

        if params["solver"] == "sgd":
            params["momentum"] = trial.suggest_float(f"{self.__class__.__name__}_momentum", *self.momentum_space)
            params["nesterovs_momentum"] = trial.suggest_categorical(f"{self.__class__.__name__}_nesterovs_momentum", self.nesterovs_momentum_space)

        params["early_stopping"] = trial.suggest_categorical(f"{self.__class__.__name__}_early_stopping", self.early_stopping_space)
        params["validation_fraction"] = trial.suggest_float(f"{self.__class__.__name__}_validation_fraction", *self.validation_fraction_space)
        params["beta_1"] = trial.suggest_float(f"{self.__class__.__name__}_beta_1", *self.beta_1_space)
        params["beta_2"] = trial.suggest_float(f"{self.__class__.__name__}_beta_2", *self.beta_2_space)
        params["epsilon"] = trial.suggest_float(f"{self.__class__.__name__}_epsilon", *self.epsilon_space)
        params["n_iter_no_change"] = trial.suggest_int(f"{self.__class__.__name__}_n_iter_no_change", *self.n_iter_no_change_space)
        params["max_fun"] = trial.suggest_int(f"{self.__class__.__name__}_max_fun", *self.max_fun_space)
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
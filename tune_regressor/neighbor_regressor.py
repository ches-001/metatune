from baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from tune_classifier import KNeighborsClassifierTuner


@dataclass
class KNeighborsRegressorTuner(KNeighborsClassifierTuner):

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        return super(KNeighborsRegressorTuner, self).sample_params(trial)

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super(KNeighborsClassifierTuner, self).sample_model(trial)
        
        params = self.sample_params(trial)
        model = super(KNeighborsClassifierTuner, self).evaluate_sampled_model("regression", KNeighborsRegressor, params)
        self.model = model

        return model


@dataclass
class RadiusNeighborsRegressorTuner(BaseTuner):
    radius_space: Iterable[int] = (2, 20)
    weight_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    leaf_size_space: Iterable[int] = (2, 100)
    p_space: Iterable[int] = (3, 10)
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan", "minkowski")
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["radius"] = trial.suggest_int(f"{self.__class__.__name__}_radius", *self.radius_space, log=False)
        params["weights"] = trial.suggest_categorical(f"{self.__class__.__name__}_weight", self.weight_space)
        params["algorithm"] = trial.suggest_categorical(f"{self.__class__.__name__}_algorithm", self.algorithm_space)
        params["leaf_size"] = trial.suggest_int(f"{self.__class__.__name__}_leaf_size", *self.leaf_size_space, log=True)
        params["p"] = trial.suggest_int(f"{self.__class__.__name__}_p", *self.p_space, log=False)
        params["metric"] = trial.suggest_categorical(f"{self.__class__.__name__}_metric", self.metric_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)

        model = super().evaluate_sampled_model("regression", RadiusNeighborsRegressor, params)

        self.model = model

        return model


tuner_model_class_dict: Dict[str, Callable] = {
    KNeighborsRegressorTuner.__name__: KNeighborsRegressor,
    RadiusNeighborsRegressorTuner.__name__: RadiusNeighborsRegressor
}
from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from tune_classifier import KNeighborsClassifierTuner


@dataclass
class KNeighborsRegressorTuner(KNeighborsClassifierTuner):

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        return super(KNeighborsRegressorTuner, self)._sample_params(trial)

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super(KNeighborsClassifierTuner, self).model(trial)
        
        params = self._sample_params(trial)
        model = super(KNeighborsClassifierTuner, self)._evaluate_sampled_model("regression", KNeighborsRegressor, params)
        self.model = model

        return model


class RadiusNeighborRegressorTuner(SampleClassMixin):
    radius_space: Iterable[int] = (1, 10)
    weight_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    leaf_size_space: Iterable[int] = (2, 60)
    p_space: Iterable[int] = (3, 10)
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan", "minkowski")

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        params["radius"] = trial.suggest_int("radius", *self.radius_space)
        params["weights"] = trial.suggest_categorical("weight", self.weight_space)
        params["algorithm"] = trial.suggest_categorical("algorithm", self.algorithm_space)
        params["leaf_size"] = trial.suggest_int("leaf_size", *self.leaf_size_space)
        params["p"] = trial.suggest_int("p", *self.p_space)
        params["metric"] = trial.suggest_categorical("metric", self.metric_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)

        model = super()._evaluate_sampled_model("regression", RadiusNeighborsRegressor, params)

        self.model = model

        return model


tuner_model_class_dict: Dict[str, Callable] = {
    KNeighborsRegressorTuner.__name__: KNeighborsRegressor,
    RadiusNeighborRegressorTuner.__name__: RadiusNeighborsRegressor
}
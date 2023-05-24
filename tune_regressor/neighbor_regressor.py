from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class KNeighborsRegressorModel(SampleClassMixin):
    n_neighbors_space: Iterable[int] = tuple(range(3, 21, 2))
    weights_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    leaf_size_space: Iterable[int] = (2, 60)
    p_space: Iterable[int] = (3, 8)
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan", "minkowski")
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        params["n_neighbors"] = trial.suggest_int("n_neighbors", *self.n_neighbors_space, log=False)
        params["weights"] = trial.suggest_categorical("weight", self.weights_space)
        params["algorithm"] = trial.suggest_categorical("algorithm", self.algorithm_space)
        params["leaf_size"] = trial.suggest_int("leaf_size", *self.leaf_size_space)
        params["p"] = trial.suggest_int("p", *self.p_space)
        params["metric"] = trial.suggest_categorical("metric", self.metric_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)

        model = super()._evaluate_sampled_model("regression", KNeighborsRegressor, params)

        self.model = model

        return model

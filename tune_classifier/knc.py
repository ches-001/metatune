from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNeighborsClassifierModel(SampleClassMixin):
    n_neighbors_space: Iterable[int] = (1, 10)
    weights_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "haversine", "manhattan", "minkowski")
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        params["n_neighbors"] = trial.suggest_int("n_neighbors", *self.n_neighbors_space, log=False)
        params["weights"] = trial.suggest_categorical("weight", self.weights_space)
        params["algorithm"] = trial.suggest_categorical("algorithm", self.algorithm_space)
        params["metric"] = trial.suggest_categorical("metric", self.metric_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)

        model = KNeighborsClassifier(
            **params
        )
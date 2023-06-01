from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from types import MappingProxyType
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid


@dataclass
class KNeighborsClassifierTuner(BaseTuner):
    n_neighbors_space: Dict[str, Any] = MappingProxyType({"low":1, "high":10, "step":2, "log":False})
    weights_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    leaf_size_space: Dict[str, Any] = MappingProxyType({"low":2, "high":100, "step":1, "log":True})
    p_space: Dict[str, Any] = MappingProxyType({"low":3, "high":8, "step":1, "log":False})
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan", "minkowski")
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["n_neighbors"] = trial.suggest_int(f"{self.__class__.__name__}_n_neighbors", **dict(self.n_neighbors_space))
        params["weights"] = trial.suggest_categorical(f"{self.__class__.__name__}_weight", self.weights_space)
        params["algorithm"] = trial.suggest_categorical(f"{self.__class__.__name__}_algorithm", self.algorithm_space)
        params["leaf_size"] = trial.suggest_int(f"{self.__class__.__name__}_leaf_size", **dict(self.leaf_size_space))
        params["p"] = trial.suggest_int(f"{self.__class__.__name__}_p", **dict(self.p_space))
        params["metric"] = trial.suggest_categorical(f"{self.__class__.__name__}_metric", self.metric_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)

        model = super().evaluate_sampled_model("classification", KNeighborsClassifier, params)

        self.model = model

        return model


@dataclass
class RadiusNeighborsClassifierTuner(BaseTuner):
    radius_space: Dict[str, Any] = MappingProxyType({"low":2, "high":20, "step":1, "log":False})
    weight_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    leaf_size_space: Dict[str, Any] = MappingProxyType({"low":2, "high":100, "step":1, "log":True})
    p_space: Dict[str, Any] = MappingProxyType({"low":3, "high":10, "step":1, "log":False})
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan", "minkowski")
    outlier_label_space: Iterable[str] = (None, "most_frequent")
    
    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["radius"] = trial.suggest_int(f"{self.__class__.__name__}_radius", **dict(self.radius_space))
        params["weights"] = trial.suggest_categorical(f"{self.__class__.__name__}_weight", self.weight_space)
        params["algorithm"] = trial.suggest_categorical(f"{self.__class__.__name__}_algorithm", self.algorithm_space)
        params["leaf_size"] = trial.suggest_int(f"{self.__class__.__name__}_leaf_size", **dict(self.leaf_size_space))
        params["p"] = trial.suggest_int(f"{self.__class__.__name__}_p", **dict(self.p_space))
        params["metric"] = trial.suggest_categorical(f"{self.__class__.__name__}_metric", self.metric_space)
        params["outlier_label"] = trial.suggest_categorical(f"{self.__class__.__name__}_outlier_label", self.outlier_label_space)

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)

        model = super().evaluate_sampled_model("classification", RadiusNeighborsClassifier, params)

        self.model = model

        return model


@dataclass
class NearestCentroidClassifierTuner(BaseTuner):
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "manhattan")
    shrink_threshold_space: Dict[str, Any] = MappingProxyType({"low":0.1, "high":0.9, "step":None, "log":False})

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        params["metric"] = trial.suggest_categorical(f"{self.__class__.__name__}_metric", self.metric_space)
        params["shrink_threshold"] = trial.suggest_float(f"{self.__class__.__name__}_shrink_threshold", **dict(self.shrink_threshold_space))

        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)

        model = super().evaluate_sampled_model("classification", NearestCentroid, params)

        self.model = model

        return model
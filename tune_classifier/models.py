from baseline import SampleClassMixin
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB


@dataclass
class SVCModel(SampleClassMixin):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    model: Any = None

    def _sample_params(self, trial: Any = None) -> Optional[Dict[str, Any]]:
        super()._sample_params(trial)

        params = {}
        params["kernel"] = trial.suggest_categorical("kernel", self.kernel_space)
        params["degree"] = trial.suggest_int("degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical("gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float("coef0", *self.coef0_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)

        return params

    def sample_model(self, trial: Any = None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = SVC(
            **params,
            class_weight="balanced",
            shrinking=True,
            probability=True)

        self.model = model
        return model


@dataclass
class KNeighborsClassifierModel(SampleClassMixin):
    n_neighbors_space: Iterable[int] = (1, 10)
    weights_space: Iterable[str] = ("uniform", "distance")
    algorithm_space: Iterable[str] = ("ball_tree", "kd_tree", "brute")
    metric_space: Iterable[str] = ("cityblock", "cosine", "euclidean", "haversine", "manhattan", "minkowski")
    model: Any = None

    def _sample_params(self, trial: Any = None) -> Optional[Dict[str, Any]]:
        super()._sample_params(trial)

        params = {}
        params["n_neighbors"] = trial.suggest_int("n_neighbors", *self.n_neighbors_space, log=False)
        params["weights"] = trial.suggest_categorical("weight", self.weights_space)
        params["algorithm"] = trial.suggest_categorical("algorithm", self.algorithm_space)
        params["metric"] = trial.suggest_categorical("metric", self.metric_space)

        return params

    def sample_model(self, trial: Any = None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)

        model = KNeighborsClassifier(
            **params
        )
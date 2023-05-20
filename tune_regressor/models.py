from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB


@dataclass
class DecisionTreeRegressorModel(SampleClassMixin):
    criterion: Iterable[str] = ("squared_error", "friedman_mse", "absolute_error", "poisson")
    splitter: Iterable[str] = ("best", "random")
    max_depth: Iterable[int] = (2, 1000)
    min_samples_split: Iterable[int] = (2, 1000)
    min_samples_leaf: Iterable[int] = (1, 1000)

    # these could also be floats, I'm not sure how to handle them
    # min_samples_split: Iterable[float] = (2, 9)
    # min_samples_leaf: Iterable[float] = (1, 9)
    
    min_weight_fraction_leaf: Iterable[float] = (0.0, 0.5)
    max_features: Iterable[str] = ("auto", "sqrt", "log2")
    max_leaf_nodes: Iterable[int] = (1, 1000)
    min_impurity_decrease: Iterable[float] = (0.0, 1.0)
    ccp_alpha: Iterable[float] = (0.0, 1.0)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["criterion"] = trial.suggest_categorical("criterion", self.criterion)
        params["splitter"] = trial.suggest_categorical("splitter", self.splitter)
        params["max_depth"] = trial.suggest_int("max_depth", *self.max_depth, log=False)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", *self.min_samples_split, log=False)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", *self.min_samples_leaf, log=False)
        params["min_weight_fraction_leaf"] = trial.suggest_float("min_weight_fraction_leaf", *self.min_weight_fraction_leaf, log=False)
        params["max_features"] = trial.suggest_categorical("max_features", self.max_features)
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", *self.max_leaf_nodes, log=False)
        params["min_impurity_decrease"] = trial.suggest_float("min_impurity_decrease", *self.min_impurity_decrease, log=False)
        params["ccp_alpha"] = trial.suggest_float("ccp_alpha", *self.ccp_alpha, log=False)
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = DecisionTreeRegressor(
            **params,)
        
        self.model = model
        return model
    

@dataclass
class SVRModel(SampleClassMixin):
    kernel_space: Iterable[str] = ("linear", "poly", "rbf", "sigmoid")
    degree_space: Iterable[int] = (1, 5)
    gamma_space: Iterable[str] = ("scale", "auto")
    coef0_space: Iterable[float] = (0.0, 0.5)
    tol_space: Iterable[float] = (1e-6, 1e-3)
    C_space: Iterable[float] = (0.9, 1.0)
    epsilon_space: Iterable[float] = (0.1, 0.5)
    model: Any = None
    
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
        super()._sample_params(trial)
        
        params = {}
        params["kernel"] = trial.suggest_categorical("kernel", self.kernel_space)
        params["degree"] = trial.suggest_int("degree", *self.degree_space, log=False)
        params["gamma"] = trial.suggest_categorical("gamma", self.gamma_space)
        params["coef0"] = trial.suggest_float("coef0", *self.coef0_space, log=False)
        params["tol"] = trial.suggest_float("tol", *self.tol_space, log=False)
        params["C"] = trial.suggest_float("C", *self.C_space, log=False)
        params["epsilon"] = trial.suggest_float("epsilon", *self.tol_space, log=False)

        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)
        
        params = self._sample_params(trial)
        model = SVR(
            **params, 
            shrinking=True)
        
        self.model = model
        return model
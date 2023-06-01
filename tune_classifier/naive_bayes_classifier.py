from ..baseline import BaseTuner
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Callable,Iterable, Optional, Dict, Any, Union
from types import MappingProxyType
from sklearn.naive_bayes import (
    BernoulliNB, 
    GaussianNB, 
    MultinomialNB, 
    ComplementNB, 
    CategoricalNB
    )

@dataclass
class GaussianNBTuner(BaseTuner):
    priors_space: Iterable[Optional[Iterable[float]]] = (None,) 
    var_smoothing_space: Dict[str, Any] = MappingProxyType({"low":1e-10, "high":1e-6, "step":None, "log":True})
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}
        
        params["priors"] = trial.suggest_categorical(f"{self.__class__.__name__}_priors", self.priors_space)
        params["var_smoothing"] = trial.suggest_float(f"{self.__class__.__name__}_var_smoothing", **dict(self.var_smoothing_space))
        
        return params

    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().sample_model(trial)

        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", GaussianNB, params)
        self.model = model
        return model


@dataclass
class BernoulliNBTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    force_alpha_space: Iterable[bool] = (True, False)
    set_binarize_space: Iterable[bool] = (True, False)
    binarize_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )    #TODO: Implement array selections

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))        
        params["force_alpha"] = trial.suggest_categorical(f"{self.__class__.__name__}_force_alpha", self.force_alpha_space)

        use_binarize = trial.suggest_categorical(f"{self.__class__.__name__}_set_binarize", self.set_binarize_space)
        if use_binarize:
            params["binarize"] = trial.suggest_float(f"{self.__class__.__name__}_binarize", **dict(self.binarize_space))
        
        params["fit_prior"] = trial.suggest_categorical("fit_prior", self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_prior", self.class_prior_space)
        
        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", BernoulliNB, params)

        self.model = model
        return model

        
@dataclass
class MultinomialNBTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})   
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))        
        params["force_alpha"]  = trial.suggest_categorical("force_alpha", self.force_alpha_space)
        params["fit_prior"] = trial.suggest_categorical("fit_prior", self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_prior", self.class_prior_space)
        
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", MultinomialNB, params)

        self.model = model
        return model

@dataclass
class ComplementNBTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )
    norm_space: Iterable[bool] = (True, False)
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["force_alpha"]  = trial.suggest_categorical("force_alpha", self.force_alpha_space)
        params["fit_prior"] = trial.suggest_categorical("fit_prior", self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_prior", self.class_prior_space)        
        params["norm"] = trial.suggest_categorical("norm", self.norm_space)
        
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", ComplementNB, params)

        self.model = model
        return model

        
@dataclass
class CategoricalNBTuner(BaseTuner):
    alpha_space: Dict[str, Any] = MappingProxyType({"low":0.0, "high":1.0, "step":None, "log":False})
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None,)
    min_categories_space: Iterable[Optional[Union[int, Iterable[int]]]] = (None,)

    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super().sample_params(trial)

        params = {}

        params["alpha"] = trial.suggest_float(f"{self.__class__.__name__}_alpha", **dict(self.alpha_space))
        params["force_alpha"]  = trial.suggest_categorical("force_alpha", self.force_alpha_space)
        params["fit_prior"] = trial.suggest_categorical("fit_prior", self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical(f"{self.__class__.__name__}_class_prior", self.class_prior_space)        
        params["min_categories"] = trial.suggest_categorical("min_categories", self.min_categories_space)    
        
        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().sample_model(trial)
        params = self.sample_params(trial)
        model = super().evaluate_sampled_model("classification", CategoricalNB, params)

        self.model = model
        return model
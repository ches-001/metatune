from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Callable,Iterable, Optional, Dict, Any, Union
from sklearn.naive_bayes import (
    BernoulliNB, 
    GaussianNB, 
    MultinomialNB, 
    ComplementNB, 
    CategoricalNB
    )

@dataclass
class GaussianNBTuner(SampleClassMixin):
    priors_space: Iterable[Optional[Iterable[float]]] = (None,) 
    var_smoothing_space: Iterable[float] = (1e-10, 1e-6)
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        
        params["priors"] = trial.suggest_categorical("priors", self.priors_space)
        params["var_smoothing"] = trial.suggest_float("var_smoothing", *self.var_smoothing_space, log=False)
        
        return params

    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", GaussianNB, params)
        self.model = model
        return model


@dataclass
class BernoulliNBTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    set_binarize_space: Iterable[bool] = (True, False)
    binarize_space: Iterable[float] = (0.0, 1.0)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )    #TODO: Implement array selections
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)        
        params['force_alpha'] = trial.suggest_categorical("force_alpha", self.force_alpha_space)

        use_binarize = trial.suggest_categorical("set_binarize", self.set_binarize_space)
        if use_binarize:
            params['binarize'] = trial.suggest_float("binarize", *self.binarize_space, log=False)
        
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical("class_prior", self.class_prior_space)
        
        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", BernoulliNB, params)

        self.model = model
        return model

        
@dataclass
class MultinomialNBTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)   
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)        
        params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical("class_prior", self.class_prior_space)
        
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", MultinomialNB, params)

        self.model = model
        return model

@dataclass
class ComplementNBTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None, )
    norm_space: Iterable[bool] = (True, False)
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical("class_prior", self.class_prior_space)        
        params['norm'] = trial.suggest_categorical('norm', self.norm_space)
        
        return params

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", ComplementNB, params)

        self.model = model
        return model

        
@dataclass
class CategoricalNBTuner(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable[float]]] = (None,)
    min_categories_space: Iterable[Optional[Union[int, Iterable[int]]]] = (None,)
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical("class_prior", self.class_prior_space)        
        params['min_categories'] = trial.suggest_categorical('min_categories', self.min_categories_space)    
        
        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", CategoricalNB, params)

        self.model = model
        return model
    
tuner_model_class_dict: Dict[str, Callable] = {
    GaussianNBTuner.__name__: GaussianNB,
    BernoulliNBTuner.__name__: BernoulliNB,
    MultinomialNBTuner.__name__: MultinomialNB,
    ComplementNBTuner.__name__: ComplementNB,
    CategoricalNBTuner.__name__: CategoricalNB,
}
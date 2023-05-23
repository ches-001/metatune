from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import (
    Iterable, 
    Optional, 
    Dict, 
    Any, 
    Union
    )

from sklearn.naive_bayes import (
    BernoulliNB, 
    GaussianNB, 
    MultinomialNB, 
    ComplementNB, 
    CategoricalNB
    )

@dataclass
class GaussianNBModel(SampleClassMixin):
    priors_space: Optional[Iterable[float]] = None  #TODO: implement array selection
    var_smoothing_space: Optional[Iterable[float]] = None
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        
        if self.priors_space is not None:
            assert sum(self.priors_space) == 1, "Sum of prior must be equal to 1" 
            params["priors"] = trial.suggest_categorical("priors", [self.priors_space])
        else:
            params['priors'] = None
        
        if self.var_smoothing_space is not None:
            params["var_smoothing"] = trial.suggest_float("var_smoothing", *self.var_smoothing_space, log=False)
        else:
            params["var_smoothing"] = 1e-9
        
        return params

    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", GaussianNB, params)
        self.model = model
        return model


@dataclass
class BernoulliNBModel(SampleClassMixin):
    alpha_space: Union[Iterable[float], int] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    binarize_space: Iterable[float] = (0.0, 1.0)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable]] = (None, )    #TODO: Implement array selections
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        if isinstance(self.alpha_space, Iterable[float]):
            params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        elif isinstance(self.alpha_space, int):
            params['alpha'] = [trial.suggest_float(*(0.0, 1.0)) for _ in range(self.alpha_space)]
        
        #params['force_alpha'] = trial.suggest_categorical("force_alpha", self.force_alpha_space)
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
class MultinomialNBModel(SampleClassMixin):
    alpha_space: Union[Iterable[float], int] = (0.0, 1.0)   
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable]] = (None, )     # TODO: Implement array selection
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        if isinstance(self.alpha_space, Iterable[float]):
            params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        elif isinstance(self.alpha_space, int):
            params['alpha'] = [trial.suggest_float(*(0.0, 1,0)) for _ in range(self.alpha_space)]
        
        #params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
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
class ComplementNBModel(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable]] = (None, )
    norm_space: Iterable[bool] = (True, False)

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        if isinstance(self.alpha_space, Iterable[float]):
            params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        elif isinstance(self.alpha_space, int):
            params['alpha'] = [trial.suggest_float(*(0.0, 1,0)) for _ in range(self.alpha_space)]

        #params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
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
class CategoricalNBModel(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Iterable[Optional[Iterable]] = (None,)
    #min_categories_space: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None
    min_category_space: Optional[Union[int, Iterable[int]]] = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        params['alpha'] = trial.suggest_float('alpha', *self.alpha_space, log=False)
        #params['force_alpha']  = trial.suggest_categorical('force_alpha', self.force_alpha_space)
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)
        params["class_prior"] = trial.suggest_categorical("class_prior", self.class_prior_space)

    
        # if isinstance(self.min_categories_space, None):
        #     if isinstance(self.min_categories_space, Iterable[int]):
        #         params['min_categories'] = trial.suggest_int('min_category', *self.min_categories_space, log=False)
        #     elif isinstance(self.min_categories_space, Iterable[Iterable]):
        #         params['min_categories'] = trial.suggest_categorical('min_category', self.min_categories_space, log=False)
        #     else:
        #         raise TypeError(f"Expected a data type of [int], or [list], but got {type(self.min_categories_space).__name__}.")
        
        if self.min_category_space is not None:
            params['min_categories'] = self.min_category_space    
        params['min_categories'] = None


        return params
    
    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super().model(trial)
        params = self._sample_params(trial)
        model = super()._evaluate_sampled_model("classification", CategoricalNB, params)

        self.model = model
        return model








# ----- NOTE ----- #
 
'''

1. for the alpha parameter the values should range from 0.0 to 1.0
   the number of features would probably have to passed from the end user

2.  for the class_prior and prior parameter, the number of features would have to be 
    passed from the end user ... if I cannot access these values from the backend here ...

3. If a use is passing Int for min_categories in categoricalNB it has to be repr in range -> (1, 100)

4. Int represents the number of features (thisi is for the alpha parameter)

'''

# ----- NOTE ----- #
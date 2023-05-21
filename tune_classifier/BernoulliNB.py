from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union
from sklearn.naive_bayes import BernoulliNB

@dataclass
class BernoulliNBModel(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    binarize_space: Iterable[float] = (0.0, 1.0)
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Optional[Iterable[float]] = None
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}
        
        params['alpha'] = trial.suggest_float("alpha", *self.alpha_space, log=False) #TODO: figure out how to implementation for array
        params['force_alpha'] = trial.suggest_categorical("force_alpha", self.force_alpha_space)
        params['binarize'] = trial.suggest_float("binarize", *self.binarize_space, log=False)
        params['fit_prior'] = trial.suggest_categorical('fit_prior', self.fit_prior_space)

        if self.class_prior_space is not None:
            assert sum(self.class_prior_space) == 1, "Sum of class_prior must be equal to 1" 
            params["class_prior"] = trial.suggest_categorical("class_prior", [self.class_prior_space])
        else:
            params['class_prior'] = None
        
        return params
    
    def sample_model(self, trial: Optional[Trial]=None) -> Any:
        super().model(trial)

        params = self._sample_params(trial)
        model = BernoulliNB(**params)

        self.model = model
        return model

        


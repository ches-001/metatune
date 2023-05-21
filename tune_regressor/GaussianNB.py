from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any
from sklearn.naive_bayes import GaussianNB

@dataclass
class GaussianNBModel(SampleClassMixin):
    priors_space: Optional[Iterable[float]] = None
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
        model = GaussianNB(
            **params
        )
        self.model = model
        return model


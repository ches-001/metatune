from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Union
from sklearn.naive_bayes import BernoulliNB

@dataclass
class BernouliNBModel(SampleClassMixin):
    alpha_space: Iterable[float] = (0.0, 1.0)
    force_alpha_space: Iterable[bool] = (True, False)
    binarize_space: Optional[float] = None
    fit_prior_space: Iterable[bool] = (True, False)
    class_prior_space: Optional[Iterable[float]] = None
    model: Any = None

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        super()._sample_params(trial)

        params = {}

        if self.alpha_space is not None:
            if isinstance(self.alpha_space, float):
                params['alpha_space'] = self.alpha_space

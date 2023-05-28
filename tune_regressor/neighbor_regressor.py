from baseline import SampleClassMixin
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Callable
from sklearn.neighbors import KNeighborsRegressor
from tune_classifier import KNeighborsClassifierTuner


@dataclass
class KNeighborsRegressorTuner(KNeighborsClassifierTuner):

    def _sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        return super(KNeighborsRegressorTuner, self)._sample_params(trial)

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super(KNeighborsClassifierTuner, self).model(trial)
        
        params = self._sample_params(trial)
        model = super(KNeighborsClassifierTuner, self)._evaluate_sampled_model("regression", KNeighborsRegressor, params)
        self.model = model

        return model


tuner_model_class_dict: Dict[str, Callable] = {
    KNeighborsRegressorTuner.__name__: KNeighborsRegressor,
}
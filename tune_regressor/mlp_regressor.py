from optuna.trial import Trial
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from sklearn.neural_network import MLPRegressor
from tune_classifier import MLPClassifierTuner


@dataclass
class MLPRegressorTuner(MLPClassifierTuner):
    
    def sample_params(self, trial: Optional[Trial] = None) -> Dict[str, Any]:
        return super(MLPRegressorTuner, self).sample_params(trial)

    def sample_model(self, trial: Optional[Trial] = None) -> Any:
        super(MLPClassifierTuner, self).sample_model(trial)

        params = self.sample_params(trial)
        model = super(MLPClassifierTuner, self).evaluate_sampled_model("regression", MLPRegressor, params)
        self.model = model
        return model


tuner_model_class_dict: Dict[str, Callable] = {
    MLPRegressorTuner.__name__: MLPRegressor
}

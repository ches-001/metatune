from optuna.trial import Trial
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SampleClassMixin:
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]: 
        if trial is None: raise ValueError("Method should be called in an optuna trial study")
    
    def model(self, trial: Optional[Trial]) -> Any:
        if trial is None: raise ValueError("Method should be called in an optuna trial study")
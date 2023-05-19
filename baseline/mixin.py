from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SampleClassMixin:
    def _sample_params(self, trial: Any=None) -> Optional[Dict[str, Any]]: 
        if trial is None: raise ValueError("Method should be called in an optuna trial study")
    
    def model(self, trial: Any) -> Any:
        if trial is None: raise ValueError("Method should be called in an optuna trial study")
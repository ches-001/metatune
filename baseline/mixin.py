import inspect, optuna
import numpy as np
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Iterable, Callable

@dataclass
class SampleClassMixin:
    def _sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]: 
        if trial is None: raise ValueError("Method should be called in an optuna trial study")
    
    def model(self, trial: Optional[Trial]) -> Any:
        if trial is None: raise ValueError("Method should be called in an optuna trial study")

    def _evaluate_params(self, model_class: Callable, params: Dict[str, Any]):
        assert isinstance(model_class, Callable), f"Invalid model_class, {model_class} is not Callable"

        param_names = list(params.keys())
        valid_param_names = list(inspect.signature(model_class.__dict__["__init__"]).parameters.keys())

        for param_name in param_names:
            if param_name not in valid_param_names:
                raise ValueError(f"invalid argument {param_name} for {model_class.__name__}")
            
    def _random_classification_set(self, is_multitask: bool=False) -> Tuple[Iterable]:
        if not is_multitask:
            return np.random.randn(20, 5), np.random.randint(0, 2, size=(20))
        
        else:
            return np.random.randn(20, 5), np.random.randint(0, 2, size=(20, 2))
    
    def _random_regression_set(self, is_multitask: bool=False) -> Tuple[Iterable]:
        if not is_multitask:
            return np.random.randn(20, 5), np.random.randn(20)
        
        else:
            return np.random.randn(20, 5), np.random.randn(20, 2)
    
    def _evaluate_sampled_model(
            self, 
            task: str, 
            model_class: Callable, 
            params: Dict[str, Any], 
            is_mulktitask: bool=False) -> Any:
        
        valid_tasks: Iterable[str] = ["regression", "classification"]
        assert task in valid_tasks, (
            f"Invalid task for self._evaluate_sampled_model, expected task to be one of {valid_tasks}, got {task}"
        )

        self._evaluate_params(model_class, params)
        
        if task == "regression":
            X, y = self._random_regression_set(is_mulktitask)
        else:
            X, y = self._random_classification_set(is_mulktitask)

        try:
            model_class(**params).fit(X, y)
        except ValueError as e:
            raise optuna.exceptions.TrialPruned(e)
        
        model = model_class(**params)
        
        return model
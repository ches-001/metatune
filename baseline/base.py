import inspect, optuna
import numpy as np
from optuna.trial import Trial
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple, Iterable, Callable
from types import MappingProxyType


class SpaceTypeValidationMixin:
    def is_space_type(self, space: Union[Dict, Iterable], type: Callable) -> bool:
        if type not in [float, int]:
            raise ValueError(
                f"is_space_type method expects type being checked to be 'int' or 'float', got {type}."
                f" To check for categorical types, try using the `is_valid_categorical_space(...)` method"
            )

        if (isinstance(space, dict) or isinstance(space, MappingProxyType)):
            valid_keys = ["low", "high"]
            for key in valid_keys:
                if key not in space.keys():
                    raise ValueError(f"defined non-categorical space is missing key-value '{key}'")
                                
            if space["low"] > space["high"]:
                raise ValueError(f"low cannot be greater than high in space: {space}")
            
            if "log" in space.keys():
                if not isinstance(space["log"], bool):
                    raise TypeError(f"log is expected to be bool type, got {type(space['log'])} instead")
                
            if "step" in space.keys():
                if space["step"] is not None:
                    if not isinstance(space["step"], float) or not isinstance(space["step"], int):
                        raise TypeError(f"step is expected to be numerical type, got {type(space['step'])} instead")
            
            return all(list(map(lambda x: isinstance(x, type), [space["low"], space["high"]])))
        
        return False
    
    def is_valid_int_space(self, space: Iterable) -> bool:
        return self.is_space_type(space, int)
    
    def is_valid_float_space(self, space: Iterable) -> bool:
        return self.is_space_type(space, float)
    
    def is_valid_categorical_space(self, space: Iterable) -> bool:
        return (not self.is_valid_float_space(space)) and (not self.is_valid_float_space(space))
            

class TrialCheckMixin:
    def in_trial(self, trial: Optional[Trial]=None) -> Dict[str, Any]: 
        if trial is None: raise ValueError("Method should be called in an optuna trial study")


class SampledModelEvaluationMixin:
    def _evaluate_params(self, model_class: Callable, params: Dict[str, Any]):
        assert isinstance(model_class, Callable), f"Invalid model_class, {model_class} is not Callable"

        param_names = list(params.keys())
        valid_param_names = list(inspect.signature(model_class.__dict__["__init__"]).parameters.keys())

        for param_name in param_names:
            if param_name not in valid_param_names:
                raise ValueError(f"invalid argument {param_name} for {model_class.__name__}")
            
    def _random_classification_set(self, is_multitask: bool=False) -> Tuple[Iterable]:
        if not is_multitask:
            return np.abs(np.random.randn(25, 5)), np.random.randint(0, 2, size=(25))
        
        else:
            return np.abs(np.random.randn(25, 5)), np.random.randint(0, 2, size=(25, 2))
    
    def _random_regression_set(self, is_multitask: bool=False) -> Tuple[Iterable]:
        if not is_multitask:
            return np.abs(np.random.randn(25, 5)), np.abs(np.random.randn(25)) + 1e-4
        
        else:
            return np.abs(np.random.randn(25, 5)), np.abs(np.random.randn(25, 2)) + 1e-4
    
    def evaluate_sampled_model(
            self, 
            task: str, 
            model_class: Callable, 
            params: Dict[str, Any], 
            is_multitask: bool=False) -> Any:
        
        valid_tasks: Iterable[str] = ["regression", "classification"]
        assert task in valid_tasks, (
            f"Invalid task for self._evaluate_sampled_model, expected task to be one of {valid_tasks}, got {task}"
        )

        self._evaluate_params(model_class, params)
        
        if task == "regression":
            X, y = self._random_regression_set(is_multitask)
        else:
            X, y = self._random_classification_set(is_multitask)

        try:
            model_class(**params).fit(X, y)
        except (ValueError, NotImplementedError) as e:
            raise optuna.exceptions.TrialPruned(e)
        
        model = model_class(**params)
        
        return model



@dataclass
class BaseTuner(SpaceTypeValidationMixin, TrialCheckMixin, SampledModelEvaluationMixin):
    r"""
    BaseTuner class that everyother tuner extends from

    If you wish to implement a custom tuner class with some default parameters, 
    you must first extend from the BaseTuner class. The custom tuner must 
    have the class attribute 'model_class' of type (Callable), which indicates
    the class of the model being tuned::
        
        from dataclasses import dataclass
        from metatune.baseline import BaseTuner
        from sklearn.gaussian_process import GaussianProcessRegressor
        from typing import Callable, Dict, Iterable, Any
        from types import MappingProxyType

        @dataclass
        class CustomTuner(BaseTuner):
            model_class: Callable = GaussianProcessRegressor
            #int space
            param1_space: Dict[str, Any] = MappingProxyType({
                "low":2, 
                "high":1000, 
                "step":1, 
                "log":True,
            })
            #float space
            param2_space: Dict[str, Any] = MappingProxyType({
                "low":0.1, 
                "high":1.0, 
                "step":None, 
                "log":False,
            })
            #categorical space
            param3_space: Iterable[str] = ("cat1", "cat2", "cat3", "cat4") -> Dict[str, Any]


            def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
                super().sample_params(trial)
                        
                params = {}
                params["param1"] = trial.suggest_int(
                    f"{self.__class__.__name__}_param1", **dict(self.param1_space))
                params["param2"] = trial.suggest_float(
                    f"{self.__class__.__name__}_param2", **dict(self.param1_space))
                params["param3"] = trial.suggest_categorical(
                    f"{self.__class__.__name__}_param3", param1_space)
                
                return params

            def sample_model(self, trial: Optional[Trial]=None) -> Any:
                super().sample_model(trial)
                params = self.sample_params(trial)

                model = super().evaluate_sampled_model(
                    "regression", self.model_class, params)

                self.model = model
                return model
    """
    model: Any = None

    def sample_params(self, trial: Trial) -> Dict[str, Any]:
        r"""
        This method sample parameters
        from the defined sample spaces and returns them. For the sake of uniqueness 
        and to avoid parameter space distribution related errors that optuna may throw,
        it is mandatory that each parameter name must be registered with a unique identifiers
        (the class name) alongside the parameter name, both seperated with an underscore (_). 
        This way, conflict is avoided when multiple tuners that happen to have same 
        model class parameter name for their corresponding model classes exist together in 
        the search space. In this method, it is mandatory to call the `super().sample_params()`
        method of the `BaseTuner` class before defining the params dictionary.

        Parameter
        ---------
        trial: optuna.trial.Trial
            optuna trial

        Return
        ------
        params: Dict[str, Any]
            collected parameters sampled from the defined search space
        """
        super().in_trial(trial)

    def sample_model(self, trial: Trial) -> Any:
        r"""
        The `sample_model(...)` method first calls the `super().sample_model()` method from the
        `BaseTuner` class. This method collects all the sampled parameters via the `sample_params(..)`
        method and passes them as argument, alongside the task --"regression" or "classification"--
        and the `model_class` into the `super().evaluate_sampled_model(...)` method. The
        `super().evaluate_sampled_model(...)` method first verifies that the keys of the params dictionary
        returned by the `sample_params(...)` method all correspond to argument names used to initialised the 
        `model_class` object, it then initialises an object of the `model_class` and fits a small set of data
        with it to check for bad sampled parameter combinations. If the combination of sampled parameters
        is bad / errotic, it triggers a pruning exception to prune the trial and move on to the next trial.
        Although It is not mandatory to use the `super().evaluate_sampled_model(...)` method in your 
        custom tuner, you are advised to use it as it handles bad parameter combination automatically 
        during search. The returned `model_class` instance after parameter and model evaluation is 
        assigned to the `model` class attribute of the `BaseTuner` class and then returned.

        Parameter
        ---------
        trial: optuna.trial.Trial
            optuna trial

        Return
        ------
        params: Any
            model initialised with sampled parameters
        """
        super().in_trial(trial)
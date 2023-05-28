import inspect, importlib
from types import ModuleType
from typing import Iterable, Tuple, Dict, Callable, Optional


def get_tuner_entities(target_module: str) -> Iterable[Tuple[str, object]]:
    module: ModuleType = importlib.import_module(target_module)
    _entities: Iterable[Tuple[str, object]] = inspect.getmembers(module)
    _classes: object = filter(lambda entity : inspect.isclass(object=entity[1]), _entities)
    tuning_entities: object = filter(lambda entity : entity[1].__module__ == target_module, _classes)
    tuning_entities: Iterable[Tuple[str, object]] = list(tuning_entities)
    tuning_entities: Iterable[Tuple[str, object]] = [(name, _class()) for name, _class in tuning_entities]

    return tuning_entities


def get_tuner_model_dict(target_module: str) -> Dict[str, Callable]:
    module: ModuleType = importlib.import_module(target_module)
    if hasattr(module, "tuner_model_class_dict"):
        return module.tuner_model_class_dict
    
    else:
        raise AttributeError(f"module {target_module} has no attribute tuner_model_class_dict")
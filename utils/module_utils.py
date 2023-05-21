import inspect
from types import ModuleType
from typing import Iterable, Tuple


def get_entities(target_module: str) -> Iterable[Tuple[str, object]]:
    module: ModuleType = __import__(target_module)
    _entities: Iterable[Tuple[str, object]] = inspect.getmembers(module)
    _classes: object = filter(lambda entity : inspect.isclass(object=entity[1]), _entities)
    tuning_entities: object = filter(lambda entity : entity[1].__module__ == target_module, _classes)
    tuning_entities: Iterable[Tuple[str, object]] = list(tuning_entities)
    tuning_entities: Iterable[Tuple[str, object]] = [(name, _class()) for name, _class in tuning_entities]

    return tuning_entities
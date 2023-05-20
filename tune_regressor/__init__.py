import inspect
from .models import *
from types import ModuleType
from typing import Iterable, Tuple, Dict

TARGET_MODULE: str = "tune_regressor.models"
module: ModuleType = __import__(TARGET_MODULE)
_entities: Iterable[Tuple[str, object]] = inspect.getmembers(module)
_classes: object = filter(lambda entity : inspect.isclass(object=entity[1]), _entities)
tuning_entities: object = filter(lambda entity : entity[1].__module__ == TARGET_MODULE, _classes)
tuning_entities: Dict = dict(tuning_entities)
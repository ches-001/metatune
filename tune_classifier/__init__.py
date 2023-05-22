from utils.module_utils import get_entities
from tune_classifier.svc import *
from tune_classifier.tree_classifier import *
from tune_classifier.linear_model_classifier import *
from tune_classifier.naive_bayes_classifier import *
from typing import Iterable, Tuple, Dict, Generator

__all__: Iterable[str] = [
    "tune_classifier.svc",
    "tune_classifier.tree_classifier",
    "tune_classifier.linear_model_classifier",
    "tune_classifier.naive_bayes_classifier",
]

tuning_entities: Iterable[Iterable[Tuple[str, object]]] = list(map(get_entities, __all__))
tuning_entities: Generator = (i for i in sum(tuning_entities, []))
from utils.module_utils import get_entities
from tune_classifier.svc import *
from tune_classifier.decision_tree_classifier import *
from tune_classifier.BernoulliNB import *
from typing import Iterable, Tuple, Dict, Generator

__all__: Iterable[str] = [
    "tune_classifier.svc",
    "tune_classifier.decision_tree_classifier",
    "tune_classifier.BernoulliNB",
]

tuning_entities: Iterable[Iterable[Tuple[str, object]]] = list(map(get_entities, __all__))
tuning_entities: Generator = (i for i in sum(tuning_entities, []))
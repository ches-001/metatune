from utils.module_utils import get_entities, get_tuner_model_dict
from tune_classifier.svc import *
from tune_classifier.tree_classifier import *
from tune_classifier.linear_model_classifier import *
from tune_classifier.ensemble_classifier import *
from tune_classifier.neighbor_classifier import *
from tune_classifier.mlp_classifier import *
from typing import Iterable, Tuple, Dict, Generator, Callable

__all__: Iterable[str] = [
    "tune_classifier.svc",
    "tune_classifier.tree_classifier",
    "tune_classifier.linear_model_classifier",
    "tune_classifier.ensemble_classifier",
    "tune_classifier.neighbor_classifier",
    "tune_classifier.mlp_classifier",
]

classifier_tuning_entities: Generator = (i for i in sum(list(map(get_entities, __all__)), []))

classifier_tuner_model_class_dict: Dict[str, Callable] = {
    k:v for _dict in map(get_tuner_model_dict, __all__) for k, v in _dict.items()
}

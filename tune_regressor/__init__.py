from utils.module_utils import get_entities, get_tuner_model_dict
from tune_regressor.svr import *
from tune_regressor.tree_regressor import *
from tune_regressor.linear_model_regressor import *
from tune_regressor.ensemble_regressor import *
from tune_regressor.neighbor_regressor import *
from tune_regressor.mlp_regressor import *
from typing import Iterable, Tuple, Dict, Generator

__all__: Iterable[str] = [
    "tune_regressor.svr",
    "tune_regressor.tree_regressor",
    "tune_regressor.linear_model_regressor",
    "tune_regressor.ensemble_regressor",
    "tune_regressor.neighbor_regressor",
    "tune_regressor.mlp_regressor",
]

regressor_tuning_entities: Generator = (i for i in sum(list(map(get_entities, __all__)), []))

regressor_tuner_model_class_dict: Dict[str, Callable] = {
    k:v for _dict in map(get_tuner_model_dict, __all__) for k, v in _dict.items()
}
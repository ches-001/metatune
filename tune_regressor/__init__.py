from utils.module_utils import get_tuner_entities, get_tuner_model_dict
from tune_regressor.svr import *
from tune_regressor.tree_regressor import *
from tune_regressor.linear_model_regressor import *
from tune_regressor.ensemble_regressor import *
from tune_regressor.neighbor_regressor import *
from tune_regressor.mlp_regressor import *
from typing import Iterable, Tuple, Dict, Generator


__modules__: Iterable[str] = [
    "tune_regressor.svr",
    "tune_regressor.tree_regressor",
    "tune_regressor.linear_model_regressor",
    "tune_regressor.ensemble_regressor",
    "tune_regressor.neighbor_regressor",
    "tune_regressor.mlp_regressor",
]

regressor_tuning_entities: Generator = (i for i in sum(list(map(get_tuner_entities, __modules__)), []))

regressor_tuner_model_class_dict: Dict[str, Callable] = {
    k:v for _dict in map(get_tuner_model_dict, __modules__) for k, v in _dict.items()
}


__all__: Iterable[str] = [
    "regressor_tuning_entities",
    "regressor_tuner_model_class_dict",
    "SVRTuner", 
    "LinearSVRTuner", 
    "NuSVRTuner", 
    "DecisionTreeRegressorTuner", 
    "ExtraTreeRegressorTuner", 
    "LinearRegressionTuner", 
    "LassoTuner", 
    "RidgeTuner", 
    "ElasticNetTuner", 
    "MultiTaskLassoTuner", 
    "MultiTaskElasticNetTuner", 
    "LarsTuner", 
    "LassoLarsTuner", 
    "LassoLarsICTuner", 
    "PassiveAggressiveRegressorTuner", 
    "QuantileRegressorTuner", 
    "SGDRegressorTuner", 
    "BayesianRidgeTuner", 
    "OrthogonalMatchingPursuitTuner", 
    "PoissonRegressorTuner", 
    "GammaRegressorTuner", 
    "TweedieRegressorTuner", 
    "RandomForestRegressorTuner", 
    "ExtraTreesRegressorTuner", 
    "AdaBoostRegressorTuner", 
    "GradientBoostingRegressorTuner", 
    "BaggingRegressorTuner", 
    "HistGradientBoostingRegressorTuner", 
    "KNeighborsRegressorTuner", 
    "MLPRegressorTuner"
]
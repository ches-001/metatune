from ..utils import make_default_tuner_type_mutable
from .svr import *
from .tree_regressor import *
from .linear_model_regressor import *
from .ensemble_regressor import *
from .neighbor_regressor import *
from .mlp_regressor import *
from typing import Iterable, Dict, Callable


regressor_tuner_model_class_map: Dict[str, Callable] = {
    SVRTuner.__name__: SVR,
    LinearSVRTuner.__name__: LinearSVR,
    NuSVRTuner.__name__: NuSVR,
    DecisionTreeRegressorTuner.__name__: DecisionTreeRegressor,
    ExtraTreeRegressorTuner.__name__: ExtraTreeRegressor,
    LinearRegressionTuner.__name__: LinearRegression,
    LassoTuner.__name__: Lasso,
    RidgeTuner.__name__: Ridge,
    ElasticNetTuner.__name__: ElasticNet,
    MultiTaskLassoTuner.__name__: MultiTaskLasso,
    MultiTaskElasticNetTuner.__name__: MultiTaskElasticNet,
    LarsTuner.__name__: Lars,
    LassoLarsTuner.__name__: LassoLars,
    LassoLarsICTuner.__name__: LassoLarsIC,
    PassiveAggressiveRegressorTuner.__name__: PassiveAggressiveRegressor,
    QuantileRegressorTuner.__name__: QuantileRegressor,
    SGDRegressorTuner.__name__: SGDRegressor,
    BayesianRidgeTuner.__name__: BayesianRidge,
    OrthogonalMatchingPursuitTuner.__name__: OrthogonalMatchingPursuit,
    PoissonRegressorTuner.__name__: PoissonRegressor,
    GammaRegressorTuner.__name__: GammaRegressor,
    TweedieRegressorTuner.__name__: TweedieRegressor,
    HuberRegressorTuner.__name__: HuberRegressor,
    TheilSenRegressorTuner.__name__: TheilSenRegressor,
    ARDRegressionTuner.__name__: ARDRegression,
    RANSACRegressorTuner.__name__: RANSACRegressor,
    RandomForestRegressorTuner.__name__: RandomForestRegressor,
    ExtraTreesRegressorTuner.__name__: ExtraTreesRegressor,
    AdaBoostRegressorTuner.__name__: AdaBoostRegressor,
    GradientBoostingRegressorTuner.__name__: GradientBoostingRegressor,
    BaggingRegressorTuner.__name__: BaggingRegressor,
    HistGradientBoostingRegressorTuner.__name__: HistGradientBoostingRegressor,
    KNeighborsRegressorTuner.__name__: KNeighborsRegressor,
    RadiusNeighborsRegressorTuner.__name__: RadiusNeighborsRegressor,
    MLPRegressorTuner.__name__: MLPRegressor,
}


regressor_search_space: Dict[str, BaseTuner] = {
    SVRTuner.__name__: SVRTuner(),
    LinearSVRTuner.__name__: LinearSVRTuner(),
    NuSVRTuner.__name__: NuSVRTuner(),
    DecisionTreeRegressorTuner.__name__: DecisionTreeRegressorTuner(),
    ExtraTreeRegressorTuner.__name__: ExtraTreeRegressorTuner(),
    LinearRegressionTuner.__name__: LinearRegressionTuner(),
    LassoTuner.__name__: LassoTuner(),
    RidgeTuner.__name__: RidgeTuner(),
    ElasticNetTuner.__name__: ElasticNetTuner(),
    MultiTaskLassoTuner.__name__: MultiTaskLassoTuner(),
    MultiTaskElasticNetTuner.__name__: MultiTaskElasticNetTuner(),
    LarsTuner.__name__: LarsTuner(),
    LassoLarsTuner.__name__: LassoLarsTuner(),
    LassoLarsICTuner.__name__: LassoLarsICTuner(),
    PassiveAggressiveRegressorTuner.__name__: PassiveAggressiveRegressorTuner(),
    QuantileRegressorTuner.__name__: QuantileRegressorTuner(),
    SGDRegressorTuner.__name__: SGDRegressorTuner(),
    BayesianRidgeTuner.__name__: BayesianRidgeTuner(),
    OrthogonalMatchingPursuitTuner.__name__: OrthogonalMatchingPursuitTuner(),
    PoissonRegressorTuner.__name__: PoissonRegressorTuner(),
    GammaRegressorTuner.__name__: GammaRegressorTuner(),
    TweedieRegressorTuner.__name__: TweedieRegressorTuner(),
    HuberRegressorTuner.__name__: HuberRegressorTuner(),
    TheilSenRegressorTuner.__name__: TheilSenRegressorTuner(),
    ARDRegressionTuner.__name__: ARDRegressionTuner(),
    RANSACRegressorTuner.__name__: RANSACRegressorTuner(),
    RandomForestRegressorTuner.__name__: RandomForestRegressorTuner(),
    ExtraTreesRegressorTuner.__name__: ExtraTreesRegressorTuner(),
    AdaBoostRegressorTuner.__name__: AdaBoostRegressorTuner(),
    GradientBoostingRegressorTuner.__name__: GradientBoostingRegressorTuner(),
    BaggingRegressorTuner.__name__: BaggingRegressorTuner(),
    HistGradientBoostingRegressorTuner.__name__: HistGradientBoostingRegressorTuner(),
    KNeighborsRegressorTuner.__name__: KNeighborsRegressorTuner(),
    RadiusNeighborsRegressorTuner.__name__: RadiusNeighborsRegressorTuner(),
    MLPRegressorTuner.__name__: MLPRegressorTuner(),
}

regressor_search_space: Dict[str, BaseTuner] = dict(
    map(lambda pair : (pair[0], make_default_tuner_type_mutable(pair[1])), regressor_search_space.items())
)



__all__: Iterable[str] = [
    "regressor_tuner_model_class_map",
    "regressor_search_space",
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
    "MLPRegressorTuner",
    "RadiusNeighborsRegressorTuner"
]
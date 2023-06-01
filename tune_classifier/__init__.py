from ..utils import make_default_tuner_type_mutable
from .svc import *
from .tree_classifier import *
from .linear_model_classifier import *
from .ensemble_classifier import *
from .naive_bayes_classifier import *
from .neighbor_classifier import *
from .mlp_classifier import *
from .discriminant_analysis_classifier import *
from typing import Iterable, Dict, Callable


classifier_tuner_model_class_map: Dict[str, Callable] = {
    SVCTuner.__name__: SVC,
    LinearSVCTuner.__name__: LinearSVC,
    NuSVCTuner.__name__: NuSVC,
    DecisionTreeClassifierTuner.__name__: DecisionTreeClassifier,
    ExtraTreeClassifierTuner.__name__: ExtraTreeClassifier,
    LogisticRegressionTuner.__name__: LogisticRegression,
    PerceptronTuner.__name__: Perceptron,
    PassiveAggressiveClassifierTuner.__name__: PassiveAggressiveClassifier,
    SGDClassifierTuner.__name__: SGDClassifier,
    RandomForestClassifierTuner.__name__: RandomForestClassifier,
    ExtraTreesClassifierTuner.__name__: ExtraTreesClassifier,
    AdaBoostClassifierTuner.__name__: AdaBoostClassifier,
    GradientBoostingClassifierTuner.__name__: GradientBoostingClassifier,
    BaggingClassifierTuner.__name__: BaggingClassifier,
    HistGradientBoostingClassifierTuner.__name__: HistGradientBoostingClassifier,
    GaussianNBTuner.__name__: GaussianNB,
    BernoulliNBTuner.__name__: BernoulliNB,
    MultinomialNBTuner.__name__: MultinomialNB,
    ComplementNBTuner.__name__: ComplementNB,
    CategoricalNBTuner.__name__: CategoricalNB,
    KNeighborsClassifierTuner.__name__: KNeighborsClassifier,
    RadiusNeighborsClassifierTuner.__name__: RadiusNeighborsClassifier,
    NearestCentroidClassifierTuner.__name__: NearestCentroid,
    MLPClassifierTuner.__name__: MLPClassifier,
    LDAClassifierTuner.__name__: LinearDiscriminantAnalysis,
    QDAClassifierTuner.__name__: QuadraticDiscriminantAnalysis,
}

classifier_search_space: Dict[str, BaseTuner] = {
    SVCTuner.__name__: SVCTuner(),
    LinearSVCTuner.__name__: LinearSVCTuner(),
    NuSVCTuner.__name__: NuSVCTuner(),
    DecisionTreeClassifierTuner.__name__: DecisionTreeClassifierTuner(),
    ExtraTreeClassifierTuner.__name__: ExtraTreeClassifierTuner(),
    LogisticRegressionTuner.__name__: LogisticRegressionTuner(),
    PerceptronTuner.__name__: PerceptronTuner(),
    PassiveAggressiveClassifierTuner.__name__: PassiveAggressiveClassifierTuner(),
    SGDClassifierTuner.__name__: SGDClassifierTuner(),
    RandomForestClassifierTuner.__name__: RandomForestClassifierTuner(),
    ExtraTreesClassifierTuner.__name__: ExtraTreesClassifierTuner(),
    AdaBoostClassifierTuner.__name__: AdaBoostClassifierTuner(),
    GradientBoostingClassifierTuner.__name__: GradientBoostingClassifierTuner(),
    BaggingClassifierTuner.__name__: BaggingClassifierTuner(),
    HistGradientBoostingClassifierTuner.__name__: HistGradientBoostingClassifierTuner(),
    GaussianNBTuner.__name__: GaussianNBTuner(),
    BernoulliNBTuner.__name__: BernoulliNBTuner(),
    MultinomialNBTuner.__name__: MultinomialNBTuner(),
    ComplementNBTuner.__name__: ComplementNBTuner(),
    CategoricalNBTuner.__name__: CategoricalNBTuner(),
    KNeighborsClassifierTuner.__name__: KNeighborsClassifierTuner(),
    RadiusNeighborsClassifierTuner.__name__: RadiusNeighborsClassifierTuner(),
    NearestCentroidClassifierTuner.__name__: NearestCentroidClassifierTuner(),
    MLPClassifierTuner.__name__: MLPClassifierTuner(),
    LDAClassifierTuner.__name__: LDAClassifierTuner(),
    QDAClassifierTuner.__name__: QDAClassifierTuner(),
}

classifier_search_space: Dict[str, BaseTuner] = dict(
    map(lambda pair : (pair[0], make_default_tuner_type_mutable(pair[1])), classifier_search_space.items())
)

__all__: Iterable[str] = [
    "classifier_tuner_model_class_map",
    "classifier_search_space",
    "SVCTuner", 
    "LinearSVCTuner", 
    "NuSVCTuner", 
    "DecisionTreeClassifierTuner", 
    "ExtraTreeClassifierTuner", 
    "LogisticRegressionTuner", 
    "PerceptronTuner", 
    "PassiveAggressiveClassifierTuner", 
    "SGDClassifierTuner", 
    "RandomForestClassifierTuner", 
    "ExtraTreesClassifierTuner", 
    "AdaBoostClassifierTuner", 
    "GradientBoostingClassifierTuner", 
    "BaggingClassifierTuner", 
    "HistGradientBoostingClassifierTuner", 
    "GaussianNBTuner", 
    "BernoulliNBTuner", 
    "MultinomialNBTuner", 
    "ComplementNBTuner", 
    "CategoricalNBTuner", 
    "KNeighborsClassifierTuner", 
    "MLPClassifierTuner",
    "RadiusNeighborsClassifierTuner",
    "NearestCentroidClassifierTuner",
    "LDAClassifierTuner",
    "QDAClassifierTuner"
]

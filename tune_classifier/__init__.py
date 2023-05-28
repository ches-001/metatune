from utils.module_utils import get_tuner_entities, get_tuner_model_dict
from tune_classifier.svc import *
from tune_classifier.tree_classifier import *
from tune_classifier.linear_model_classifier import *
from tune_classifier.ensemble_classifier import *
from tune_classifier.naive_bayes_classifier import *
from tune_classifier.neighbor_classifier import *
from tune_classifier.mlp_classifier import *
from typing import Iterable, Dict, Generator, Callable


__modules__: Iterable[str] = [
    "tune_classifier.svc",
    "tune_classifier.tree_classifier",
    "tune_classifier.linear_model_classifier",
    "tune_classifier.ensemble_classifier",
    "tune_classifier.naive_bayes_classifier",
    "tune_classifier.neighbor_classifier",
    "tune_classifier.mlp_classifier",
]

classifier_tuning_entities: Generator = (i for i in sum(list(map(get_tuner_entities, __modules__)), []))

classifier_tuner_model_class_dict: Dict[str, Callable] = {
    k:v for _dict in map(get_tuner_model_dict, __modules__) for k, v in _dict.items()
}

__all__: Iterable[str] = [
    "classifier_tuning_entities",
    "classifier_tuner_model_class_dict",
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
    "NearestCentroidClassifierTuner"
]

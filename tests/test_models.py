from baseline.mixin import SampleClassMixin
import tune_classifier
import tune_regressor
from tests.utils import BaseTest


# run python -m pytest -v


class TestSVR(BaseTest):
    model: SampleClassMixin = tune_regressor.SVRModel
    task: str = "regression"


class TestDecisionTreeRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.DecisionTreeRegressorModel
    task: str = "regression"


class TestSVC(BaseTest):
    model: SampleClassMixin = tune_classifier.SVCModel
    task: str = "classification"


class TestDecisionTreeClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.DecisionTreeClassifierModel
    task: str = "classification"

class TestBernoulliNBModel(BaseTest):
    model: SampleClassMixin = tune_classifier.BernoulliNBModel
    task: str = "classification"

class TestGaussianNBModel(BaseTest):
    model: SampleClassMixin = tune_classifier.GaussianNBModel
    task: str = "classification"

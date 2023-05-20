from baseline.mixin import SampleClassMixin
from tune_regressor import *
from tune_classifier import *
from tests.utils import BaseTest


# run python -m pytest -v


class TestSVR(BaseTest):
    model: SampleClassMixin = SVRModel
    task: str = "regression"


class TestDecisionTreeRegressor(BaseTest):
    model: SampleClassMixin = DecisionTreeRegressorModel
    task: str = "regression"


class TestSVC(BaseTest):
    model: SampleClassMixin = SVCModel
    task: str = "classification"


class TestDecisionTreeClassifier(BaseTest):
    model: SampleClassMixin = DecisionTreeClassifierModel
    task: str = "classification"


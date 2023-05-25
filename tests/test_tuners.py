from baseline.mixin import SampleClassMixin
import tune_classifier
import tune_regressor
from tests.utils import BaseTest


# run python -m pytest -v


class TestSVR(BaseTest):
    model: SampleClassMixin = tune_regressor.SVRTuner()
    task: str = "regression"


class TestDecisionTreeRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.DecisionTreeRegressorTuner()
    task: str = "regression"


class TestSVC(BaseTest):
    model: SampleClassMixin = tune_classifier.SVCTuner()
    task: str = "classification"


class TestDecisionTreeClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.DecisionTreeClassifierTuner()
    task: str = "classification"


class TestLinearRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.LinearRegressionTuner()
    task: str = "regression"


class TestLassoRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.LassoTuner()
    task: str = "regression"


class TestRidgeRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.RidgeTuner()
    task: str = "regression"


class TestLogisticRegressor(BaseTest):
    model: SampleClassMixin = tune_classifier.LogisticRegressionTuner()
    task: str = "classification"


class TestLinearSVC(BaseTest):
    model: SampleClassMixin = tune_classifier.LinearSVCTuner()
    task: str = "classification"


class TestLinearSVR(BaseTest):
    model: SampleClassMixin = tune_regressor.LinearSVRTuner()
    task: str = "regression"


class TestExtraTreeClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.ExtraTreeClassifierTuner()
    task: str = "classification"


class TestExtraTreeRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.ExtraTreeRegressorTuner()
    task: str = "regression"


class TestRandomForestClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.RandomForestClassifierTuner()
    task: str = "classification"


class TestRandomForestRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.RandomForestRegressorTuner()
    task: str = "regression"


class TestExtraTreesClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.ExtraTreesClassifierTuner()
    task: str = "classification"


class TestExtraTreesRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.ExtraTreesRegressorTuner()
    task: str = "regression"


class TestExtraTreeClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.ExtraTreeClassifierTuner()
    task: str = "classification"


class TestExtraTreeRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.ExtraTreeRegressorTuner()
    task: str = "regression"


class TestAdaBoostClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.AdaBoostClassifierTuner()
    task: str = "classification"

    
class TestAdaBoostRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.AdaBoostRegressorTuner()
    task: str = "regression"


class TestKNNClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.KNeighborsClassifierTuner()
    task: str = "classification"


class TestKNNRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.KNeighborsRegressorTuner()
    task: str = "regression"


class TestElasticNetRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.ElasticNetTuner()
    task: str = "regression"


class TestMultiTaskLassoRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.MultiTaskLassoTuner()
    task: str = "regression"


class TestMultiTaskElasticNetRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.MultiTaskElasticNetTuner()
    task: str = "regression"


class TestBaggingClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.BaggingClassifierTuner()
    task: str = "classification"


class TestBaggingRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.BaggingRegressorTuner()
    task: str = "regression"


class TestGradientBoostingClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.GradientBoostingClassifierTuner()
    task: str = "classification"


class TestGradientBoostingRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.GradientBoostingRegressorTuner()
    task: str = "regression"


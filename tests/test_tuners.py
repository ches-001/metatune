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


class TestNuSVC(BaseTest):
    model: SampleClassMixin = tune_classifier.NuSVCTuner()
    task: str = "classification"


class TestNuSVR(BaseTest):
    model: SampleClassMixin = tune_regressor.NuSVRTuner()
    task: str = "regression"


class TestPerceptron(BaseTest):
    model: SampleClassMixin = tune_classifier.PerceptronTuner()
    task: str = "classification"


class TestPassiveAggressiveClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.PassiveAggressiveClassifierTuner()
    task: str = "classification"


class TestPassiveAggressiveRegressorTuner(BaseTest):
    model: SampleClassMixin = tune_regressor.PassiveAggressiveRegressorTuner()
    task: str = "regression"


class TestSGDClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.SGDClassifierTuner()
    task: str = "classification"


class TestSGDRegressorTuner(BaseTest):
    model: SampleClassMixin = tune_regressor.SGDRegressorTuner()
    task: str = "regression"


class TestMLPClassifierTuner(BaseTest):
    model: SampleClassMixin = tune_classifier.MLPClassifierTuner()
    task: str = "classification"
    n_trials: int = 10
    

class TestMLPRegressorTuner(BaseTest):
    model: SampleClassMixin = tune_regressor.MLPRegressorTuner()
    task: str = "regression"
    n_trials: int = 10

      
class TestHistGradientBoostingClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.HistGradientBoostingClassifierTuner()
    task: str = "classification"


class TestHistGradientBoostingRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.HistGradientBoostingRegressorTuner()
    task: str = "regression"

class TestLars(BaseTest):
    model: SampleClassMixin = tune_regressor.LarsTuner()
    task: str = "regression"


class TestLassoLars(BaseTest):
    model: SampleClassMixin = tune_regressor.LassoLarsTuner()
    task: str = "regression"


class TestLassoLarsIC(BaseTest):
    model: SampleClassMixin = tune_regressor.LassoLarsICTuner()
    task: str = "regression"

class TestBayesianRidge(BaseTest):
    model: SampleClassMixin = tune_regressor.BayesianRidgeTuner()
    task: str = "regression"

class TestGaussianNBClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.GaussianNBTuner()
    task: str = "classification"

class TestBernoulliNBClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.BernoulliNBTuner()
    task: str = "classification"

class TestMultinomialNBClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.MultinomialNBTuner()
    task: str = "classification"

class TestComplementNBClassifier(BaseTest):
    model: SampleClassMixin = tune_classifier.ComplementNBTuner()
    task: str = "classification"

class TestCategoricalNBTuner(BaseTest):
    model: SampleClassMixin = tune_classifier.CategoricalNBTuner()
    task: str = "classification"

class TestTweedieRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.TweedieRegressorTuner()
    task: str = "regression"


class TestOrthogonalMatchingPursuit(BaseTest):
    model: SampleClassMixin = tune_regressor.OrthogonalMatchingPursuitTuner()
    task: str = "regression"


class TestPoissonRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.PoissonRegressorTuner()
    task: str = "regression"


class TestGammaRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.GammaRegressorTuner()
    task: str = "regression"


class TestQuantileRegressor(BaseTest):
    model: SampleClassMixin = tune_regressor.QuantileRegressorTuner()
    task: str = "regression"
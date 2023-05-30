from baseline.base import BaseTuner
import tune_classifier
import tune_regressor
from tests.utils import BaseTest


# run python -m pytest -v


class TestSVR(BaseTest):
    model: BaseTuner = tune_regressor.SVRTuner()
    task: str = "regression"


class TestDecisionTreeRegressor(BaseTest):
    model: BaseTuner = tune_regressor.DecisionTreeRegressorTuner()
    task: str = "regression"


class TestSVC(BaseTest):
    model: BaseTuner = tune_classifier.SVCTuner()
    task: str = "classification"


class TestDecisionTreeClassifier(BaseTest):
    model: BaseTuner = tune_classifier.DecisionTreeClassifierTuner()
    task: str = "classification"


class TestLinearRegressor(BaseTest):
    model: BaseTuner = tune_regressor.LinearRegressionTuner()
    task: str = "regression"


class TestLassoRegressor(BaseTest):
    model: BaseTuner = tune_regressor.LassoTuner()
    task: str = "regression"


class TestRidgeRegressor(BaseTest):
    model: BaseTuner = tune_regressor.RidgeTuner()
    task: str = "regression"


class TestLogisticRegressor(BaseTest):
    model: BaseTuner = tune_classifier.LogisticRegressionTuner()
    task: str = "classification"


class TestLinearSVC(BaseTest):
    model: BaseTuner = tune_classifier.LinearSVCTuner()
    task: str = "classification"


class TestLinearSVR(BaseTest):
    model: BaseTuner = tune_regressor.LinearSVRTuner()
    task: str = "regression"


class TestExtraTreeClassifier(BaseTest):
    model: BaseTuner = tune_classifier.ExtraTreeClassifierTuner()
    task: str = "classification"


class TestExtraTreeRegressor(BaseTest):
    model: BaseTuner = tune_regressor.ExtraTreeRegressorTuner()
    task: str = "regression"


class TestRandomForestClassifier(BaseTest):
    model: BaseTuner = tune_classifier.RandomForestClassifierTuner()
    task: str = "classification"


class TestRandomForestRegressor(BaseTest):
    model: BaseTuner = tune_regressor.RandomForestRegressorTuner()
    task: str = "regression"


class TestExtraTreesClassifier(BaseTest):
    model: BaseTuner = tune_classifier.ExtraTreesClassifierTuner()
    task: str = "classification"


class TestExtraTreesRegressor(BaseTest):
    model: BaseTuner = tune_regressor.ExtraTreesRegressorTuner()
    task: str = "regression"


class TestExtraTreeClassifier(BaseTest):
    model: BaseTuner = tune_classifier.ExtraTreeClassifierTuner()
    task: str = "classification"


class TestExtraTreeRegressor(BaseTest):
    model: BaseTuner = tune_regressor.ExtraTreeRegressorTuner()
    task: str = "regression"


class TestAdaBoostClassifier(BaseTest):
    model: BaseTuner = tune_classifier.AdaBoostClassifierTuner()
    task: str = "classification"

    
class TestAdaBoostRegressor(BaseTest):
    model: BaseTuner = tune_regressor.AdaBoostRegressorTuner()
    task: str = "regression"


class TestKNNClassifier(BaseTest):
    model: BaseTuner = tune_classifier.KNeighborsClassifierTuner()
    task: str = "classification"


class TestKNNRegressor(BaseTest):
    model: BaseTuner = tune_regressor.KNeighborsRegressorTuner()
    task: str = "regression"


class TestNearestCentroidClassifier(BaseTest):
    model: BaseTuner = tune_classifier.NearestCentroidClassifierTuner()
    task: str = "classification"


class TestElasticNetRegressor(BaseTest):
    model: BaseTuner = tune_regressor.ElasticNetTuner()
    task: str = "regression"


class TestMultiTaskLassoRegressor(BaseTest):
    model: BaseTuner = tune_regressor.MultiTaskLassoTuner()
    task: str = "regression"


class TestMultiTaskElasticNetRegressor(BaseTest):
    model: BaseTuner = tune_regressor.MultiTaskElasticNetTuner()
    task: str = "regression"


class TestBaggingClassifier(BaseTest):
    model: BaseTuner = tune_classifier.BaggingClassifierTuner()
    task: str = "classification"


class TestBaggingRegressor(BaseTest):
    model: BaseTuner = tune_regressor.BaggingRegressorTuner()
    task: str = "regression"


class TestGradientBoostingClassifier(BaseTest):
    model: BaseTuner = tune_classifier.GradientBoostingClassifierTuner()
    task: str = "classification"


class TestGradientBoostingRegressor(BaseTest):
    model: BaseTuner = tune_regressor.GradientBoostingRegressorTuner()
    task: str = "regression"


class TestRadiusNeighborClassifier(BaseTest):
    model: BaseTuner = tune_classifier.RadiusNeighborsClassifierTuner()
    task: str = "classification"


class TestRadiusNeighborRegressor(BaseTest):
    model: BaseTuner = tune_regressor.RadiusNeighborsRegressorTuner()
    task: str = "regression"


class TestNuSVC(BaseTest):
    model: BaseTuner = tune_classifier.NuSVCTuner()
    task: str = "classification"


class TestNuSVR(BaseTest):
    model: BaseTuner = tune_regressor.NuSVRTuner()
    task: str = "regression"


class TestPerceptron(BaseTest):
    model: BaseTuner = tune_classifier.PerceptronTuner()
    task: str = "classification"


class TestPassiveAggressiveClassifier(BaseTest):
    model: BaseTuner = tune_classifier.PassiveAggressiveClassifierTuner()
    task: str = "classification"


class TestPassiveAggressiveRegressorTuner(BaseTest):
    model: BaseTuner = tune_regressor.PassiveAggressiveRegressorTuner()
    task: str = "regression"


class TestSGDClassifier(BaseTest):
    model: BaseTuner = tune_classifier.SGDClassifierTuner()
    task: str = "classification"


class TestSGDRegressorTuner(BaseTest):
    model: BaseTuner = tune_regressor.SGDRegressorTuner()
    task: str = "regression"


class TestMLPClassifierTuner(BaseTest):
    model: BaseTuner = tune_classifier.MLPClassifierTuner()
    task: str = "classification"
    n_trials: int = 10
    

class TestMLPRegressorTuner(BaseTest):
    model: BaseTuner = tune_regressor.MLPRegressorTuner()
    task: str = "regression"
    n_trials: int = 10

      
class TestHistGradientBoostingClassifier(BaseTest):
    model: BaseTuner = tune_classifier.HistGradientBoostingClassifierTuner()
    task: str = "classification"


class TestHistGradientBoostingRegressor(BaseTest):
    model: BaseTuner = tune_regressor.HistGradientBoostingRegressorTuner()
    task: str = "regression"

class TestLars(BaseTest):
    model: BaseTuner = tune_regressor.LarsTuner()
    task: str = "regression"


class TestLassoLars(BaseTest):
    model: BaseTuner = tune_regressor.LassoLarsTuner()
    task: str = "regression"


class TestLassoLarsIC(BaseTest):
    model: BaseTuner = tune_regressor.LassoLarsICTuner()
    task: str = "regression"

class TestBayesianRidge(BaseTest):
    model: BaseTuner = tune_regressor.BayesianRidgeTuner()
    task: str = "regression"

class TestGaussianNBClassifier(BaseTest):
    model: BaseTuner = tune_classifier.GaussianNBTuner()
    task: str = "classification"

class TestBernoulliNBClassifier(BaseTest):
    model: BaseTuner = tune_classifier.BernoulliNBTuner()
    task: str = "classification"

class TestMultinomialNBClassifier(BaseTest):
    model: BaseTuner = tune_classifier.MultinomialNBTuner()
    task: str = "classification"

class TestComplementNBClassifier(BaseTest):
    model: BaseTuner = tune_classifier.ComplementNBTuner()
    task: str = "classification"

class TestCategoricalNBTuner(BaseTest):
    model: BaseTuner = tune_classifier.CategoricalNBTuner()
    task: str = "classification"

class TestTweedieRegressor(BaseTest):
    model: BaseTuner = tune_regressor.TweedieRegressorTuner()
    task: str = "regression"


class TestOrthogonalMatchingPursuit(BaseTest):
    model: BaseTuner = tune_regressor.OrthogonalMatchingPursuitTuner()
    task: str = "regression"


class TestPoissonRegressor(BaseTest):
    model: BaseTuner = tune_regressor.PoissonRegressorTuner()
    task: str = "regression"


class TestGammaRegressor(BaseTest):
    model: BaseTuner = tune_regressor.GammaRegressorTuner()
    task: str = "regression"


class TestQuantileRegressor(BaseTest):
    model: BaseTuner = tune_regressor.QuantileRegressorTuner()
    task: str = "regression"


class TestHuberRegressor(BaseTest):
    model: BaseTuner = tune_regressor.HuberRegressorTuner()
    task: str = "regression"


class TestTheilSenRegressor(BaseTest):
    model: BaseTuner = tune_regressor.TheilSenRegressorTuner()
    task: str = "regression"


class TestARDRegressor(BaseTest):
    model: BaseTuner = tune_regressor.ARDRegressionTuner()
    task: str = "regression"


class TestRANSACRegressor(BaseTest):
    model: BaseTuner = tune_regressor.RANSACRegressorTuner()
    task: str = "regression"


class TestLinearDiscriminantAnalysis(BaseTest):
    model: BaseTuner = tune_classifier.LDAClassifierTuner()
    task: str = "classification"


class TestQuadraticDiscriminantAnalysis(BaseTest):
    model: BaseTuner = tune_classifier.QDAClassifierTuner()
    task: str = "classification"

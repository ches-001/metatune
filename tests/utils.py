import optuna
import sklearn
from sklearn import datasets
from optuna.trial import Trial
from baseline.mixin import SampleClassMixin


# load sample datasets
# regression
REG_X, REG_y = datasets.load_diabetes(return_X_y=True)
REG_DATA = sklearn.model_selection.train_test_split(REG_X, REG_y, random_state=0)

# classification
CLS_X, CLS_y = datasets.load_iris(return_X_y=True)
CLS_DATA = sklearn.model_selection.train_test_split(CLS_X, CLS_y, random_state=0)


# Define an objective function to be minimized.
def objective_factory(model: SampleClassMixin, task: str="regression"):
    """ generate objective function for given model """
    # unpack data
    if task == "regression":
        data = REG_DATA
        metric = sklearn.metrics.mean_squared_error
    else:
        data = CLS_DATA
        metric = sklearn.metrics.accuracy_score

    X_train, X_val, y_train, y_val = data
    def objective(trial: Trial):
        sampled_model = model().sample_model(trial)
        sampled_model.fit(X_train, y_train)
        y_pred = sampled_model.predict(X_val)
        error = metric(y_val, y_pred)
        return error
    return objective


class BaseTest:
    model: SampleClassMixin = None
    task: str = None

    def test_methods(self):
        assert hasattr(self.model, "_sample_params")
        assert hasattr(self.model, "sample_model")

    def test_study(self):
        try:
            study = optuna.create_study()  # Create a new study.
            study.optimize(objective_factory(self.model, task=self.task), n_trials=20)
            best = study.best_params
        except:
            best = None
        assert best is not None





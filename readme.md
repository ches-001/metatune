# MetaTune

MetaTune framework implements simultaneous model selection and hyperparameter tuning of sci-kit-learn supervised learning algorithms with [optuna](https://github.com/optuna/optuna) implemented metaheuristic search algorithms

<br>
<br>
<hr>

## Getting Started
<hr>
MetaTune implements user customizable tuners for all sci-kit-learn supervised learning algorithms, with it one can easily sampled the best model out of a search space of all sc-kit-learn algorithms and their corresponding hyperparameters with relative with only a few lines of code.

Importing the `MetaTune(...)` class is as easy as:

```python
from metatune import MetaTune

metatune = MetaTune(task="classification")
```

We can further use this object to select the machine learning algorithm and the corresponding parameters that best model our data like so:

```python
import optuna
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


# load classification dataset
X, y = datasets.load_iris(return_X_y=True)

# define data scaler
scaler = MinMaxScaler()

# split data to training and evaluation set
X_train, X_eval, y_train, y_eval = train_test_split(X, y, random_state=0, shuffle=True)

#scaler features
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)


def objective(trial):
    model = metatune.sample_models_with_params(trial)
    model.fit(X_train, y_train)

    pred = model.predict(X_eval)
    f1 = f1_score(y_eval, pred, average="micro")
    recall = recall_score(y_eval, pred, average="micro")
    precision = precision_score(y_eval, pred, average="micro")

    return (f1 + recall + precision) / 3


# create optuna study
study = optuna.create_study(
    study_name="classifier_model",
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(), # pruner is necessary to prune bad parameter combinations of models when sampling
    sampler=optuna.samplers.TPESampler(),
)

study.optimize(objective, n_trials=50)

#output
...
# [I 2023-06-01 07:19:01,678] Trial 27 finished with value: 0.8947368421052632 and parameters: {'model_tuner': 'AdaBoostClassifierTuner', #'AdaBoostClassifierTuner_estimator': None, 'AdaBoostClassifierTuner_n_estimators': 90, 'AdaBoostClassifierTuner_learning_rate': #              0.12391614619885151, 'AdaBoostClassifierTuner_algorithm': 'SAMME.R', 'AdaBoostClassifierTuner_random_state': 7705}. Best is trial 1 with value: # 0.9736842105263158.
...
```

In this task, we hope to find the best classification algorithm and its hyperparameters that best maximize objective, which happens to be the average sum of f1, recall and precision metrics.

After running this, we can retrieve the best optuna trial and build a model out of it for further finetuning with more data, like so:

```python
sampled_model = metatune.build_sampled_model(study.best_trial)
```
**NOTE** that the models returned are purely sci-kit-learn models, thus the reason the regular `fit(...)` and `predict(...)` methods can be called on them.

<br>

**NOTE:** A thing to note is that for large datasets, hyperparameter search ought to be used on a subset of the data rather than the entire dataset. The reason for this being that the aim of hyperparameter tuning is not fit all of the data, but rather to sample the parameters that best model the data. After tuning, you can instantiate the sampled model with the corresponding sampled parameters and fine-tune the model on the large dataset, this way you avoid the unnecessary computation required to model the entirety of the available dataset.

<br>
<br>
<hr>

## Handling Model Data Incompatibility
<hr>
In some cases, not all sci-kit-learn models will be compatible with your data. In such cases, you can do one of two things

<br>

### 1. Calling the `only_compatible_with_data(...)` method:
This method takes in three arguments, X, y and probability_score. X and y correspond to the training data, this attempts a data fit on all models (intialised with there default parameters) in the search space, it exempts models that are not compatible to the dataset, hence reducing the search space.

**NOTE:**, Some models may not actually be incompatible with your dataset, and a model may be exempted simply because the default parameter being used raises an exception when an attempt to fit the model on the data is made. so it may not be suitable to use this method in some cases, hence the reason to opt for the second option

The `probability_score` (default=False) when set to `True` remove models that do not implement the `predict_proba(...)` method, this can be handy if you wish to maximise an objective that relies on the predicted probability scores for each class rather than predicted labels. This argument is only effective is `task = "classification"`.

**NOTE:**, Some models may not have the `predict_proba(...)` readily available until certain conditions are met in the parameter combination used to initialise the model object. For example, the `SVC` class objects will only have this method if the `probability` parameter is set to `True`.

<br>

### 2. raising an optuna.exceptions.TrialPruned(e)
Calling `.fit(...)` on your dataset may throw an exception regardless of whether the combination of sampled parameters is correct, for example, calling the `.fit(...)` method of a naive bayes classifier model on dataset with non-positive features will raise an exception, or in some cases, the sampled `nu` parameter for the `NuSVC` or `NuSVR` model may not be compatible with your dataset and may raise an exception. In cases like these the optimization process will be terminated due to these unforseen errors. To handle them, you will need to catch the exception and instead, raise a `TrialPruned` exceptions instead. You can do this like so:

```python
...
try:
    model.fit(X_train, y_train)

except Exception as e:
    optuna.exceptions.TrialPruned(e)
```

This way, instead of terminating the metaheuristic search program, optuna simply skips to the next trial and gives you a log relating to why the trial was pruned. Similarly, if you happen to want to optimize an objective the relies on predicted probability scores for each class, rather than the class label, you can call the `predict_proba(...)` method on `X_train`  in the `try` block, just after called the `fit(...)` method.

<br>
<br>
<hr>

## Advanced Configurations

In the `MetaTuner(..)` class, there are four more arguments alongside the `task` argument used for advanced configuration when initialising a `MetaTuner(..)` class, namely: `custom_tuners`, `custom_only`, `excluded` and `single_tuner`.
<br>

### 1. custom_tuners Argument: ```Optional[Iterable[BaseTuner]] (Default=None)```

The `custom_tuners` argument is used to specify custom tuners or overwrite the default parameter space of existing tuners in the sample space. If you wish to extend or shrink the sample space of a given parameter in tuner, you can do so as shown:

```python
from metatune import MetaTune
from metatune.tune_classifier import NuSVCTuner, SGDClassifierTuner

nu_svc_tuner = NuSVCTuner(
    nu_space={"low":0.1, "high":0.4, "step":None, "log":False}
)

sgd_classifier_tuner = SGDClassifierTuner(
    loss_space=('hinge', 'log_loss', 'modified_huber', 'squared_hinge')
)

metatune = MetaTune(task="classification", custom_tuners=[nu_svc_tuner, sgd_classifier_tuner])
```

You can view all the tuners in the search space with the `search_space` attribute of the `MataTune` instance like so:

```python
metatune.search_space
```
<br>

### 2. custom_only Argument: ```bool (Default=False)```
If set to `True`, this argument makes it such that only the list of tuners specified in the `custom_tuners` argument make up the search space
<br>

### 3. excluded: ```Optional[Iterable[Union[str, Callable]]] (default=None)```
This argument is used to specify a list of tuners to exempt from the search space. It takes as value, a list of strings or Callable types corresponding to class names or class of tuners being excluded, it can be used as shown:

```python
from metatune.tune_classifier import NuSVCTuner

metatune = MetaTune(task="classification", excluded=[NuSVCTuner, "SGDClassifierTuner"])
```
<br>

### 4. single_tuner: ```Optional[BaseTuner] (default=None)```
This is used when you wish to perform parameter searching for a single specific model class with its corresponding tuner. If specified, only the tuner passed as argument will exit in the search space. You can use this to specify a custom tuner or an already implemented tuner with or without overwriting the parameter search spaces as shown:

```python

from metatune.tune_classifier import GradientBoostingClassifierTuner

metatune = MetaTune(task="classification", single_tuner=GradientBoostingClassifierTuner())
```
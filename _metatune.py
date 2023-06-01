import inspect, copy
from .baseline import BaseTuner, TrialCheckMixin
from optuna.trial import Trial, FrozenTrial
from .tune_regressor import regressor_search_space, regressor_tuner_model_class_map
from .tune_classifier import classifier_search_space, classifier_tuner_model_class_map
from .utils import make_default_tuner_type_mutable
from typing import Iterable, Tuple, Dict, Union, Optional, Any, Callable


class MetaTune(TrialCheckMixin):

    r"""
        This class implements a sample utility for model and hyperparameter 
        sampling from customizable space

        Parameters
        ----------
        task : str
            Specifies the data modeling task 'regression' or 'classification'

        custom_tuners: Optional[Iterable[BaseTuner]], default=None
            Iterable of user defined tuners. The tuners in this list can either 
            be custom made by the user or one of the already existing tuners 
            available in this framework. This argument is especially useful if 
            you wish to change the default hyperparameter space of a given tuner,
            you can import the tuner class from the designated module and overwite
            the default hyperparameter space, this way, during hyperparameter 
            sampling, the sampler searchings through the custom space instead of 
            the default space::

                from metatune.tune_classifier import NuSVCTuner
                nusvc_tuner = NuSVCTuner(nu_space={"low":0.2, "high":1.0, "step":None, "log":False})
                MetaTune(task="regression", custom_tuners=[nusvc_tuner])

            If you wish to implement a custom tuner class with some default parameters, 
            you must first extend from the BaseTuner class. The custom tuner must 
            have the class attribute 'model_class' of type (Callable), which indicates
            the class of the model being tuned::
                
                from dataclasses import dataclass
                from metatune.baseline import BaseTuner
                from sklearn.gaussian_process import GaussianProcessRegressor
                from typing import Callable, Dict, Iterable, Any
                from types import MappingProxyType

                @dataclass
                class CustomTuner(BaseTuner):
                    model_class: Callable = GaussianProcessRegressor
                    #int space
                    param1_space: Dict[str, Any] = MappingProxyType({
                        "low":2, 
                        "high":1000, 
                        "step":1, 
                        "log":True,
                    })
                    #float space
                    param2_space: Dict[str, Any] = MappingProxyType({
                        "low":0.1, 
                        "high":1.0, 
                        "step":None, 
                        "log":False,
                    })
                    #categorical space
                    param3_space: Iterable[str] = ("cat1", "cat2", "cat3", "cat4") -> Dict[str, Any]


                    def sample_params(self, trial: Optional[Trial]=None) -> Dict[str, Any]:
                        super().sample_params(trial)
                                
                        params = {}
                        params["param1"] = trial.suggest_int(
                            f"{self.__class__.__name__}_param1", **dict(self.param1_space))
                        params["param2"] = trial.suggest_float(
                            f"{self.__class__.__name__}_param2", **dict(self.param1_space))
                        params["param3"] = trial.suggest_categorical(
                            f"{self.__class__.__name__}_param3", param1_space)
                        
                        return params

                    def sample_model(self, trial: Optional[Trial]=None) -> Any:
                        super().sample_model(trial)
                        params = self.sample_params(trial)

                        model = super().evaluate_sampled_model(
                            "regression", self.model_class, params)

                        self.model = model
                        return model

                MetaTune(task="regression", custom_tuners=[CustomTuner()])

            Every custom tuner shoud be initialised with the `@dataclass` decorator and
            must extend from the `BaseTuner` class.
            Since a dataclass require that its default class attributes are immutable, 
            we cannot directly use a dictionary to assign values of an int or float
            sample space, this is because dictionaries are mutable data structures, so
            in this case we use the `MappingProxyType` class to wrap any dictionary, 
            hence extending immutable characteristics, which can them be used to 
            initialise an integer or float space.

            For the sake of naming conventions, all class attributes that represent a
            sample space of some sort, must be named after the corresponding parameter
            in the model with an '_space' suffix. 

            The tuner has two methods, visit the BaseTuner class documentation to read up
            about them as they are mandatory for defining a custom tuner.


        excluded : Optional[Iterable[Union[str, Callable]]], default=None
            An iterable of str or callable type that specifies the list of tuners 
            of a given task to be exempted. This is especially useful if you have 
            identified beforehand that some models are not compatible with your 
            dataset.

        custom_only : bool, default=False
            Specifies if only custom tuners should be used for model and 
            hyperparameter sampling. This argument only applies if `custom_tuners`
            is specified.

        single_tuner: Optional[BaseTuner], default=None
            If specified, only one tuner is used through out the sampling process, 
            inotherwords the sampling algorithm will only be sampling hyperparameters
            for the specific tuner and no other.
    """

    def __init__(
            self, 
            task: str,
            custom_tuners: Optional[Iterable[BaseTuner]]=None, 
            excluded: Optional[Iterable[Union[str, Callable]]]=None,
            custom_only: bool=False, 
            single_tuner: Optional[BaseTuner]=None):
        
        valid_tasks: Iterable[str] = ["classification", "regression"]
        if task not in valid_tasks:
            raise ValueError(
                f"Invalid task {task}, expects tasks to be 'regression' or 'classification', got {task}")
        
        self.task = task
        self.custom_tuners = custom_tuners
        self.excluded = excluded
        self.custom_only = custom_only
        self.single_tuner = single_tuner

        if self.task == "regression":
            self.search_space: Dict[str, BaseTuner] = copy.deepcopy(regressor_search_space)
            self.tuner_model_class_map: Dict[str, Callable] = copy.deepcopy(regressor_tuner_model_class_map)

        else:
            self.search_space:  Dict[str, BaseTuner] = copy.deepcopy(classifier_search_space)
            self.tuner_model_class_map: Dict[str, Callable] = copy.deepcopy(classifier_tuner_model_class_map)

        self._exclude_tuners()
        self._prepare_custom_tuners()

        if self.single_tuner is not None:
            self.search_space, self.tuner_model_class_map = self._get_single_tuner(self.single_tuner)


    def _exclude_tuners(self):
        if self.excluded is None: 
            return 
        
        for tuner_class in self.excluded:
            if isinstance(tuner_class, Callable):
                key = tuner_class.__name__

            elif isinstance(tuner_class, str):
                key = tuner_class

            else:
                raise ValueError(
                    "items of 'excluded' must either be of type str or Callable,"
                    " corresponding to the class name or class of defined tuner to be excluded,"
                    f" got {type(tuner_class)} instead")
            
            if key in self.search_space.keys():
                self.search_space.pop(key)

            if key in self.tuner_model_class_map.keys():
                self.tuner_model_class_map.pop(key)


    def _prepare_custom_tuners(self):
        if self.custom_tuners is None:
            return
        
        _search_space: Dict[str, BaseTuner] = {}
        _tuner_model_class_map: Dict[str, Callable] = {}

        for tuner in self.custom_tuners:
            space, map_dict = self._get_single_tuner(tuner)
            _search_space.update(space)
            _tuner_model_class_map.update(map_dict)

        if self.custom_only:
            self.search_space: Dict[str, BaseTuner] = _search_space
            self.tuner_model_class_map: Dict[str, Callable] = _tuner_model_class_map

        else:
            self.search_space: Dict[str, BaseTuner] = {**_search_space, **self.search_space}
            self.tuner_model_class_map: Dict[str, Callable] = {**_tuner_model_class_map, **self.tuner_model_class_map}

        
    def _get_single_tuner(self, tuner: BaseTuner) -> Tuple[Dict[str, BaseTuner], Dict[str, Callable]]:
        if tuner is None:
            return
        
        # by default some tuner attributes are of MappingProxyType objects. This wrapper was essential
        # for sustaining the immutable nature of default class attributes of dataclasses, however we
        # want to make these types mutable once more to avoid implementation issues.
        tuner = make_default_tuner_type_mutable(tuner)

        if not isinstance(tuner, BaseTuner):
            raise ValueError(f"{tuner} most be of type or extend from {BaseTuner}")
        
        _search_space: Dict[str, BaseTuner] = {tuner.__class__.__name__: tuner}

        # check if tuner object name exists in the default self.tuner_model_class_map, 
        # if it does, it is a good indication that the custom tuner (BaseTuner type) is
        # on that already exists in the system, whose default space parameters were 
        # probably edited by the user.
        if tuner.__class__.__name__ in self.tuner_model_class_map.keys():
            _tuner_model_class_map = {
                tuner.__class__.__name__ : self.tuner_model_class_map[tuner.__class__.__name__]
            }

        # if tuner object name does not exist in the self.tuner_model_class_map, it indicates
        # that the tuner (baseTuner type) is a custom tuner that is not part of the library of
        # tuners in this framework. This tuner is expected to have a 'model_class' (Callable type) 
        # attribute which corresponds to the class of the model being tuned.
        else:
            if not hasattr(tuner, "model_class"):
                raise AttributeError(
                    F"{tuner.__class__.__name__}() has no attribute 'model_class', which corresponds"
                    " to the class implementation of the tuned model")
            
            if not isinstance(getattr(tuner, "model_class"), Callable):
                    raise TypeError(
                        f"'model_class' attribute of {tuner.__class__.__name__}() is expected to be of type"
                        f" Callable, got {type(getattr(tuner, 'model_class'))}")
            
            _tuner_model_class_map = {tuner.__class__.__name__: getattr(tuner, "model_class")}

        return _search_space, _tuner_model_class_map
            

    def only_compatible_with_data(self, X: Iterable, y: Iterable, probability_score: bool=False) -> Iterable[str]:
        r"""
        This method checks the tuners in the search space that are incompatible 
        with the given data, and automatically exludes them from the search space
        and tuner model map. This way, during optuna sampling, less trials are 
        pruned

        Note: 
            Calling this method is compulsory, as it may filter out tuners whose
            corresponding models are compatible with your data, but have default
            parameters that may lead to exceptions.
        
        Parameters
        ----------
        X : Iterable | Array like of shape (n_samples, n_features)
            Feature vectors or other representations of training data,
            (preferably preprocessed)

        y : Iterable | Array like of shape (n_samples, ) or (n_samples, labels)
            Target values to predict, (preferably preprocessed)

        probability_score : bool, default=True
            use the `predict_proba(...)` method of the tuner `model_class` to
            verify if `model_class` can output probability scores. Only useful
            if `self.task="classification"`

        Return
        ------
        tuners: Iterable[str]
            name of tuners that have been excluded due to incompatibility with data
        """
        
        excluded = []
        _tuner_names = list(self.tuner_model_class_map.keys())

        for tuner_name in _tuner_names:
            _model = self.tuner_model_class_map[tuner_name]()
            
            if not hasattr(_model, "fit"):
                raise AttributeError(
                    f"{_model} does not have method 'fit(...)'. This method is crucial and must be implemented"
                    " in the model_class of your custom tuner")
            
            if not hasattr(_model, "predict"):
                raise AttributeError(
                    f"{_model} does not have method 'predict(...)'. This method is crucial and must be implemented"
                    " in the model_class of your custom tuner")
            
            try:
                _model.fit(X, y)
                _model.predict(X)

                if probability_score:
                    if self.task == "classification" and not hasattr(_model, "predict_proba"):
                        raise AttributeError()

            except Exception as e:
                if tuner_name in self.search_space.keys():
                    self.search_space.pop(tuner_name)

                if tuner_name in self.tuner_model_class_map.keys():
                    self.tuner_model_class_map.pop(tuner_name)

                excluded.append(tuner_name)

        if len(_tuner_names) == len(excluded):
            raise Exception(
                "No tuner seems to be compatible with your data, ensure that your datatypes and format"
                " are correct, no NaN values are present and all column vectors are numerical")

        return excluded
    

    def sample_models_with_params(self, trial: Trial) -> Any:
        r"""
            This method samples a tuner corresponding to a model and samples the 
            corresponding hyperparameters from the search space defined in the tuner
            that best optimizes an objective.

            Parameters
            ----------
            trial : optuna.trial.Trial
                optuna trial

            Return
            ------
            model: Any
                sampled model object initialised with sampled hyperparameters. 
                Note: model must implement `fit(...)` method.
        """

        super().in_trial(trial)
        tuner_name: str = trial.suggest_categorical("model_tuner", list(self.search_space.keys()))
        tuner: BaseTuner = self.search_space[tuner_name]
        model = tuner.sample_model(trial)

        return model


    def build_sampled_model(self, best_trial: FrozenTrial, **kwargs) -> Any:
        r"""
            This method initialises a model corresponding to the sampled
            tuner from the corresponding sampled parameters of the best 
            trial in the optuna study.

            Parameters
            ----------
            best_trial : optuna.trial.FrozenTrial
                best trial of the optuna study

            kwargs (optional):
                arguments corresponding model class of selected tuner

            Return
            ------
            model: Any
                sampled model object initialised with sampled hyperparameters. 
                Note: model must implement `fit(...)` method.
        """

        tuner_name: str = best_trial.params["model_tuner"]
        model_class = self.tuner_model_class_map[tuner_name]

        model_params_names = list(inspect.signature(model_class.__dict__["__init__"]).parameters.keys())
        best_params_dict = {
            k.replace(f"{tuner_name}_", "") : v 
            for k, v in best_trial.params.items() 
            if k.replace(f"{tuner_name}_", "") in model_params_names
            }

        params = {**kwargs, **best_params_dict}
        return model_class(**params)
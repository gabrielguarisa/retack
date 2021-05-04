import os.path
from typing import Any, Callable, Dict, List, Type, Union

import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.utils import import_element, load_elements, unique_name


class Experiment(object):
    def __init__(
        self,
        models: Union[Dict[str, BaseEstimator], List[BaseEstimator]],
        metric_funcs: List[Callable],
        cv_method: BaseCrossValidator,
        n_jobs: int = None,
        X=None,
        y=None,
    ):
        self._models = {}
        if isinstance(models, list):
            for model in models:
                model_name = unique_name(
                    model.__class__.__name__, list(self._models.keys())
                )
                self._models[model_name] = model
        elif isinstance(models, dict):
            self._models = models
        else:
            raise TypeError("Models must be a list or a dictionary!")
        self._cv_method = cv_method
        self._metric_funcs = metric_funcs
        self._n_jobs = n_jobs
        self._results = None

        if X is not None and y is not None:
            self.__call__(X, y)

    @property
    def models(self) -> Dict[str, BaseEstimator]:
        return self._models

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    def __call__(self, X, y) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            y_pred = cross_val_predict(
                model, X, y, cv=self._cv_method, n_jobs=self._n_jobs
            )
            results.append([name] + [f(y, y_pred) for f in self._metric_funcs])

        cols = ["model"] + [f.__name__ for f in self._metric_funcs]

        self._results = pd.DataFrame(results, columns=cols).set_index(
            keys="model"
        )
        return self._results


class ExperimentManager(object):
    def __init__(
        self,
        models: List[Type[BaseEstimator]],
        metric_funcs: List[Callable],
        *,
        model_names: List[str] = None,
        model_args: List[Dict[str, Any]] = None,
    ):
        if len(models) == 0:
            raise ValueError("The number of models must be greater than zero!")

        if len(metric_funcs) == 0:
            raise ValueError(
                "The number of metric_funcs must be greater than zero!"
            )

        if model_args is None:
            model_args = [{} for _ in range(len(models))]
        if model_names is None:
            model_names = []
            for i in range(len(models)):
                model_names.append(
                    unique_name(models[i].__class__.__name__, model_names)
                )
        elif len(model_args) != len(models) or len(model_names) != len(models):
            raise ValueError("models and model_args must be the same lenght!")

        self._models = models
        self._metric_funcs = metric_funcs
        self._model_args = model_args
        self._model_names = model_names

    @property
    def models(self) -> List[Type[BaseEstimator]]:
        return self._models

    @property
    def metric_funcs(self) -> List[Callable]:
        return self._metric_funcs

    @property
    def model_args(self) -> List[Dict[str, Any]]:
        return self._model_args

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    def to_dict(self) -> Dict[str, List[Any]]:
        return {
            "models": self.models,
            "metric_funcs": self.metric_funcs,
            "model_names": self.model_names,
            "model_args": self.model_args,
        }

    @classmethod
    def load(cls, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"File {filename} (or the relevant path) does not exist."
            )

        with open(filename, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        models = load_elements(data["models"])

        return cls(
            models=models["elements"],
            model_args=models["args"],
            model_names=models["names"],
            metric_funcs=[
                import_element(name) for name in data.get("metrics", [])
            ],
        )

    def run(
        self, X, y, cv_method: BaseCrossValidator = KFold(), n_jobs: int = None
    ):
        return Experiment(
            models={
                self.model_names[i]: self.models[i](**self.model_args[i])
                for i in range(len(self.models))
            },
            metric_funcs=self.metric_funcs,
            cv_method=cv_method,
            n_jobs=n_jobs,
            X=X,
            y=y,
        )

import os.path
from pydoc import locate
from typing import Callable, Dict, List, Type, Union

import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator


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
            for m in models:
                model_name = m.__class__.__name__
                i = 1
                while model_name in self._models:
                    model_name = f"{m.__class__.__name__}_{i}"
                    i += 1
                else:
                    self._models[model_name] = m
        elif isinstance(models, dict):
            self._models = models
        else:
            raise TypeError("Models must be a list or a dictionary!")
        self._cv_method = cv_method
        self._metric_funcs = metric_funcs
        self._n_jobs = n_jobs
        self._metrics = None

        if X is not None and y is not None:
            self.__call__(X, y)

    @property
    def models(self) -> Dict[str, BaseEstimator]:
        return self._models

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics

    def __call__(self, X, y) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            y_pred = cross_val_predict(
                model, X, y, cv=self._cv_method, n_jobs=self._n_jobs
            )
            results.append([name] + [f(y, y_pred) for f in self._metric_funcs])

        cols = ["model"] + [f.__name__ for f in self._metric_funcs]

        self._metrics = pd.DataFrame(results, columns=cols).set_index(
            keys="model"
        )
        return self._metrics


class ExperimentManager(object):
    def __init__(
        self, models: List[Type[BaseEstimator]], metric_funcs: List[Callable]
    ):
        if len(models) == 0:
            raise ValueError("The number of models must be greater than zero!")

        if len(metric_funcs) == 0:
            raise ValueError(
                "The number of metric_funcs must be greater than zero!"
            )

        self._models = models
        self._metric_funcs = metric_funcs

    @property
    def models(self) -> List[Type[BaseEstimator]]:
        return self._models

    @property
    def metric_funcs(self) -> List[Callable]:
        return self._metric_funcs

    @staticmethod
    def _import_element(name: str) -> object:
        elem = locate(name)
        if elem is None:
            raise ImportError(f"Failed to import {name}")
        return elem

    @classmethod
    def load(cls, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"File {filename} (or the relevant path) does not exist."
            )
        with open(filename, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return cls(
            models=[
                cls._import_element(name) for name in data.get("models", [])
            ],
            metric_funcs=[
                cls._import_element(name) for name in data.get("metrics", [])
            ],
        )

    def run(
        self, X, y, cv_method: BaseCrossValidator = KFold(), n_jobs: int = None
    ):
        return Experiment(
            models=[m() for m in self.models],
            metric_funcs=self.metric_funcs,
            cv_method=cv_method,
            n_jobs=n_jobs,
            X=X,
            y=y,
        )

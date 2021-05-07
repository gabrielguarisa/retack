from typing import Callable, Dict, List, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.base import ExperimentBase
from retack.utils import get_element_name, get_instance_or_class, unique_name


class Experiment(ExperimentBase):
    def __init__(
        self,
        models: Union[
            Dict[str, BaseEstimator],
            Dict[str, Type[BaseEstimator]],
            List[BaseEstimator],
            List[Type[BaseEstimator]],
        ],
        metric_funcs: List[Callable],
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
    ):
        self.set_models(models)
        self._metric_funcs = metric_funcs
        self._results = None
        super().__init__(cv_method=cv_method, n_jobs=n_jobs)

    def set_models(
        self,
        models: Union[
            Dict[str, BaseEstimator],
            Dict[str, Type[BaseEstimator]],
            List[BaseEstimator],
            List[Type[BaseEstimator]],
        ],
    ):
        if len(models) == 0:
            raise ValueError("The number of models must be greater than zero!")

        self._models = {}
        if isinstance(models, list) or isinstance(models, np.ndarray):
            for i in range(len(models)):
                name = unique_name(
                    get_element_name(models[i]),
                    list(self._models.keys()),
                )
                self._models[name] = get_instance_or_class(models[i])
        elif isinstance(models, dict):
            self._models = {
                name: get_instance_or_class(model)
                for name, model in models.items()
            }
        else:
            raise TypeError("Invalid type for models!")

    @property
    def models(self) -> Dict[str, BaseEstimator]:
        return self._models

    @property
    def metric_funcs(self) -> List[BaseEstimator]:
        return self._metric_funcs

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    def run(self, X, y, **kwargs) -> pd.DataFrame:
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

from typing import Any, Callable, Dict, Type

import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.base import Optimizer


class OptunaOptimizer(Optimizer):
    def __init__(
        self,
        model: Type[BaseEstimator],
        model_args: Dict[str, Any],
        metric_func: Callable,
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
        **kwargs,
    ):
        super().__init__(model, model_args, metric_func, cv_method, n_jobs)
        self.new_study(**kwargs)
        self._X = None
        self._y = None

    def new_study(self, **kwargs):
        self._study = optuna.create_study(**kwargs)
        return self.study

    @property
    def study(self):
        return self._study

    def __call__(self, trial):
        model_params = {}
        for key, value in self._model_args.items():
            model_params[key] = getattr(trial, value["type"])(
                key, **value["args"]
            )

        y_pred = cross_val_predict(
            self._model(**model_params),
            self._X,
            self._y,
            cv=self._cv_method,
            n_jobs=self._n_jobs,
        )

        self._results.append(self._metric_func(self._y, y_pred))

        return self._results[-1]

    def run(self, X, y, **kwargs):
        self._results = []
        self._X = X
        self._y = y
        self._study.optimize(self.__call__, **kwargs)
        return self._study.best_params, self._study.best_value

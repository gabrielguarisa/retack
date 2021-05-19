import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Tuple, Type, Union

import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.base import Optimizer


class OptunaOptimizer(Optimizer):
    def __init__(
        self,
        model: Union[Type[BaseEstimator], BaseEstimator],
        model_args: Dict[str, Any],
        metric_func: Callable,
        metric_func_args: Dict[str, Any] = {},
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
        **kwargs,
    ):
        super().__init__(
            model, model_args, metric_func, metric_func_args, cv_method, n_jobs
        )
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

        self._results.append(
            self.metric_func(self._y, y_pred, **self.metric_func_args)
        )

        return self._results[-1]

    def run(self, X, y, n_jobs=None, **kwargs) -> Tuple[Dict[str, Any], float]:
        self._results = []
        self._X = X
        self._y = y

        if n_jobs is None or n_jobs == 1:
            self._study.optimize(self.__call__, **kwargs)
        elif n_jobs == -1 or n_jobs > 1:
            n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for _ in range(n_jobs):
                    executor.submit(
                        self._study.optimize, self.__call__, **kwargs
                    )
        else:
            raise ValueError("Invalid n_jobs value")

        return self._study.best_params, self._study.best_value

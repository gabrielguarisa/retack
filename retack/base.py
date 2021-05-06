from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection._split import BaseCrossValidator


class ExperimentBase(ABC):
    def __init__(
        self, cv_method: BaseCrossValidator = KFold(), n_jobs: int = None
    ):
        self._cv_method = cv_method
        self._n_jobs = n_jobs

    @abstractmethod
    def run(self, X, y, **kwargs):
        pass


class Optimizer(ExperimentBase):
    def __init__(
        self,
        model: Type[BaseEstimator],
        model_args: Dict[str, Any],
        metric_func: Callable,
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
    ):
        self._model = model
        self._model_args = model_args
        self._metric_func = metric_func

        self._results = None
        super().__init__(cv_method, n_jobs)

    @property
    def results(self) -> List[float]:
        return self._results

    @property
    def model(self) -> Type[BaseEstimator]:
        return self._model

    @property
    def model_args(self) -> Dict[str, Any]:
        return self._model_args

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection._split import BaseCrossValidator

from retack.utils import get_instance_or_class


class ExperimentBase(ABC):
    def __init__(
        self,
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
        verbose: bool = False,
    ):
        self._cv_method = cv_method
        self._n_jobs = n_jobs
        self._verbose = verbose

    @abstractmethod
    def run(self, X, y, **kwargs):  # pragma: no cover
        pass


class Optimizer(ExperimentBase):
    def __init__(
        self,
        model: Union[Type[BaseEstimator], BaseEstimator],
        model_args: Dict[str, Any],
        metric_func: Callable,
        metric_func_args: Dict[str, Any] = {},
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
    ):
        self._model = get_instance_or_class(model, return_instance=False)
        self._model_args = model_args
        self._metric_func = metric_func
        self._metric_func_args = metric_func_args

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

    @property
    def metric_func(self) -> Callable:
        return self._metric_func

    @property
    def metric_func_args(self) -> Dict[str, Any]:
        return self._metric_func_args

from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.base import ExperimentBase, Optimizer
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
        metric_funcs: Union[List[Callable], Dict[str, Callable]],
        metric_funcs_args: Union[
            List[Dict[str, Any]], Dict[str, Dict[str, Any]]
        ] = None,
        cv_method: BaseCrossValidator = KFold(),
        n_jobs: int = None,
    ):
        self._models = {}
        self._metric_funcs = {}
        self._optimizers = {}
        self.set_models(models)
        self.set_metric_funcs(metric_funcs, metric_funcs_args)
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

    def set_metric_funcs(
        self,
        metric_funcs: Union[List[Callable], Dict[str, Callable]],
        metric_funcs_args: Union[
            List[Dict[str, Any]], Dict[str, Dict[str, Any]]
        ] = {},
    ):
        if (
            type(metric_funcs) != type(metric_funcs_args)
            and metric_funcs_args is not None
        ):
            raise TypeError(
                "metric_funcs and metric_funcs_args must have the same type"
            )
        elif metric_funcs_args is None:
            metric_funcs_args = (
                {}
                if isinstance(metric_funcs, dict)
                else [{} for _ in range(len(metric_funcs))]
            )

        self._metric_funcs = {}
        self._metric_funcs_args = {}
        if isinstance(metric_funcs, list) or isinstance(
            metric_funcs, np.ndarray
        ):
            if len(metric_funcs) != len(metric_funcs_args):
                raise ValueError(
                    "metric_funcs and metric_funcs_args have different sizes"
                )
            for i in range(len(metric_funcs)):
                name = unique_name(
                    get_element_name(metric_funcs[i]),
                    list(self._models.keys()),
                )
                self._metric_funcs[name] = metric_funcs[i]
                self._metric_funcs_args[name] = metric_funcs_args[i]
        elif isinstance(metric_funcs, dict):
            for name, metric_func in metric_funcs.items():
                self._metric_funcs[name] = metric_func
                self._metric_funcs_args[name] = metric_funcs_args.get(name, {})
        else:
            raise TypeError("Invalid type for metric_funcs")

    def set_optimizer(
        self,
        model_name: str,
        optimizer: Union[Optimizer, Type[Optimizer]],
        metric_func: Callable,
        model_args: Dict[str, Any] = {},
        use_optimizer_args: bool = True,
        **kwargs,
    ):
        if model_name not in self.models:
            raise ValueError(f"{model_name} model not found")

        if isinstance(optimizer, Optimizer) and use_optimizer_args:
            model_args = optimizer.model_args

        self._optimizers[model_name] = get_instance_or_class(
            optimizer, return_instance=False
        )(
            model=self.models[model_name],
            model_args=model_args,
            metric_func=metric_func,
            **kwargs,
        )

    @property
    def models(self) -> Dict[str, BaseEstimator]:
        return self._models

    @property
    def metric_funcs(self) -> Dict[str, Callable]:
        return self._metric_funcs

    @property
    def metric_funcs_args(self) -> Dict[str, Dict[str, Callable]]:
        return self._metric_funcs_args

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    @property
    def optimizers(self) -> Dict[str, Optimizer]:
        return self._optimizers

    def run(self, X, y, **kwargs) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            y_pred = cross_val_predict(
                model, X, y, cv=self._cv_method, n_jobs=self._n_jobs
            )
            results.append(
                [name]
                + [
                    func(y, y_pred, **self.metric_funcs_args[name])
                    for name, func in self.metric_funcs.items()
                ]
            )

        cols = ["model"] + list(self.metric_funcs.keys())

        self._results = pd.DataFrame(results, columns=cols).set_index(
            keys="model"
        )
        return self._results

    def optimize(
        self, model_name: str, X, y, return_model: bool = False, **kwargs
    ) -> Union[
        Tuple[Dict[str, Any], float],
        Tuple[Dict[str, Any], float, BaseEstimator],
    ]:
        if model_name not in self.optimizers:
            raise ValueError(f"{model_name} optimizer not found")
        best_params, best_value = self.optimizers[model_name].run(
            X, y, **kwargs
        )
        if return_model:
            return (
                best_params,
                best_value,
                get_instance_or_class(
                    self.models[model_name], return_instance=False
                )(**best_params),
            )
        return best_params, best_value

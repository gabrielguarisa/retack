import os.path
from typing import Callable, Dict, List, Union

import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection._split import BaseCrossValidator

from retack.utils import import_element, load_elements, set_params_to_dict


class Experiment(object):
    def __init__(
        self,
        models: Union[Dict[str, BaseEstimator], List[BaseEstimator]],
        metric_funcs: List[Callable],
    ):
        self._models = set_params_to_dict(models, "models")
        self._metric_funcs = metric_funcs
        self._results = None

    @property
    def models(self) -> Dict[str, BaseEstimator]:
        return self._models

    @property
    def metric_funcs(self) -> List[BaseEstimator]:
        return self._metric_funcs

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    def run(
        self, X, y, cv_method: BaseCrossValidator = KFold(), n_jobs: int = None
    ) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            y_pred = cross_val_predict(
                model, X, y, cv=cv_method, n_jobs=n_jobs
            )
            results.append([name] + [f(y, y_pred) for f in self._metric_funcs])

        cols = ["model"] + [f.__name__ for f in self._metric_funcs]

        self._results = pd.DataFrame(results, columns=cols).set_index(
            keys="model"
        )
        return self._results

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
            models={
                models["names"][i]: models["elements"][i](**models["args"][i])
                for i in range(len(models["elements"]))
            },
            metric_funcs=[
                import_element(name) for name in data.get("metrics", [])
            ],
        )

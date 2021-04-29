import os.path
from pydoc import locate
from typing import List, Type

import yaml
from sklearn.base import BaseEstimator


class ExperimentManager(object):
    def __init__(self, models: List[Type[BaseEstimator]]):
        if len(models) == 0:
            raise ValueError(
                "The number of models must be greater than zero !"
            )
        self._models = models

    @property
    def models(self) -> List[Type[BaseEstimator]]:
        return self._models

    @staticmethod
    def _load_models(names: List[str]) -> List[Type[BaseEstimator]]:
        models = []
        for model_name in names:
            model = locate(model_name)
            if model is None:
                raise ImportError(f"Failed to load model: {model_name}")

            models.append(model)

        return models

    @classmethod
    def load(cls, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(
                f"File {filename} (or the relevant path) does not exist."
            )
        with open(filename, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return cls(models=cls._load_models(data.get("models", [])))

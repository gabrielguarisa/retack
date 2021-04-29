from pydoc import locate
from typing import List, Type

import yaml
from sklearn.base import BaseEstimator


class ExperimentManager(object):
    def __init__(self, models: List[Type[BaseEstimator]]):
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

        return models

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return cls(models=cls._load_models(data["models"]))
